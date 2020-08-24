import logging
from datetime import datetime

import apache_beam as beam
import numpy as np
import pandas as pd
import tensorflow as tf
from apache_beam.metrics import Metrics
from tensorflow.python.framework import ops

import tensorflow_transform as tft

from .config import (DATA_FEATURE_SPEC, ONE_DIM_FLOAT_FEATURES,
                     ONE_DIM_INT_FEATURES, SCALAR_FLOAT_FEATURES,
                     SCALAR_INT_FEATURES, SCALAR_STRING_FEATURES, SEQ_LEN)


class BQQuery(object):
    train_query = """
SELECT
  -- static values
  item_number,
  ANY_VALUE(item_description) AS item_description,
  ANY_VALUE(cost) AS cost,
  ANY_VALUE(retail) AS retail,
  ANY_VALUE(volume_ml) AS volume_ml,
  ANY_VALUE(pack_size) AS pack_size,
  -- time series values
  year_month,
  ARRAY_AGG(date ORDER BY date) AS dates_arr,
  ARRAY_AGG(daily_bottles_sold ORDER BY date) AS daily_bottles_sold_arr
FROM (
  SELECT
    -- static values
    item_number,
    ANY_VALUE(item_description) AS item_description,
    AVG(state_bottle_cost) AS cost,
    AVG(state_bottle_retail) AS retail,
    AVG(bottle_volume_ml) AS volume_ml,
    AVG(pack) AS pack_size,
    -- time series values
    FORMAT_DATE('%Y-%m', date) AS year_month,
    date,
    SUM(bottles_sold) AS daily_bottles_sold
  FROM `bigquery-public-data.iowa_liquor_sales.sales`
  GROUP BY item_number, date
  )
GROUP BY item_number, year_month
    """
    predict_query = """
SELECT
  -- static values
  item_number,
  ANY_VALUE(item_description) AS item_description,
  ANY_VALUE(cost) AS cost,
  ANY_VALUE(retail) AS retail,
  ANY_VALUE(volume_ml) AS volume_ml,
  ANY_VALUE(pack_size) AS pack_size,
  -- time series values
  year_month,
  ARRAY_AGG(date ORDER BY date) AS dates_arr,
  ARRAY_AGG(daily_bottles_sold ORDER BY date) AS daily_bottles_sold_arr
FROM (
  SELECT
    -- static values
    item_number,
    ANY_VALUE(item_description) AS item_description,
    AVG(state_bottle_cost) AS cost,
    AVG(state_bottle_retail) AS retail,
    AVG(bottle_volume_ml) AS volume_ml,
    AVG(pack) AS pack_size,
    -- time series values
    FORMAT_DATE('%Y-%m', date) AS year_month,
    date,
    SUM(bottles_sold) AS daily_bottles_sold
  FROM (
      SELECT *
      FROM `bigquery-public-data.iowa_liquor_sales.sales`
      WHERE date BETWEEN "2020-07-01" AND "2020-07-15"
      )
  GROUP BY item_number, date
  )
GROUP BY item_number, year_month
    """


class MapAndFilterErrors(beam.PTransform):
    """Like beam.Map but filters out errors in the map_fn."""
    class _MapAndFilterErrorsDoFn(beam.DoFn):
        """Count the bad examples using a beam metric."""
        def __init__(self, fn):
            self._fn = fn
            self._elements_counter = Metrics.counter(self._fn.__name__, 'n_elements')
            # Create a counter to measure number of bad elements.
            self._bad_elements_counter = Metrics.counter(self._fn.__name__, 'failed_elements')
            # major python builtin exceptions
            self._assert_err = Metrics.counter(self._fn.__name__, 'AssertionErrors')
            self._key_err = Metrics.counter(self._fn.__name__, 'KeyErrors')
            self._type_err = Metrics.counter(self._fn.__name__, 'TypeErrors')
            self._val_err = Metrics.counter(self._fn.__name__, 'ValueErrors')
            self._other_err = Metrics.counter(self._fn.__name__, 'other_errors')

        def process(self, element):
            try:
                results = self._fn(element)
                # if multiple elements are returned from 1 element - yield 1 at a time
                if isinstance(results, list):
                    for result in results:
                        yield result
                # if 1 element is returned from 1 element - output the single result
                else:
                    yield results
            except Exception as e:  # pylint: disable=broad-except
                logging.error("%s failed for %s due to unhandled exception %s",
                              self._fn.__name__, element, e, exc_info=True)
                # Catch any exception the above call.
                self._bad_elements_counter.inc(1)
                if isinstance(e, AssertionError):
                    self._assert_err.inc(1)
                elif isinstance(e, KeyError):
                    self._key_err.inc(1)
                elif isinstance(e, TypeError):
                    self._type_err.inc(1)
                elif isinstance(e, ValueError):
                    self._val_err.inc(1)
                else:
                    self._other_err.inc(1)
            finally:
                self._elements_counter.inc(1)

    def __init__(self, fn):
        self._fn = fn

    def expand(self, pcoll):
        return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))


def prepare_seq_sample(raw_sample):
    """
    Prepare 1 sequence sample through appropriate padding, etc.

    Args:
        raw_sample (dict): 1 row from BigQuery result

    Returns:
        dict: matches specified TFRecord schema

    """
    year, month = raw_sample["year_month"].split('-')
    prepared_sample = {
        'item_id': int(raw_sample['item_number']),
        'year': int(year),
        # 0 index
        'month': int(month) - 1
    }
    for feature in SCALAR_STRING_FEATURES:
        prepared_sample[feature] = raw_sample[feature]
    for feature in SCALAR_INT_FEATURES:
        if feature not in prepared_sample:
            prepared_sample[feature] = int(raw_sample.get(feature, -1))
    for feature in SCALAR_FLOAT_FEATURES:
        prepared_sample[feature] = float(raw_sample[feature])
    df = pd.DataFrame(
        {'daily_bottles_sold': np.array(raw_sample['daily_bottles_sold_arr'], dtype=int),
         'date': pd.to_datetime(raw_sample['dates_arr']),
         'valid_day': True})
    df["cumu_bottles_sold"] = df["daily_bottles_sold"].cumsum()
    df["total_bottles_sold"] = df["daily_bottles_sold"].sum()
    df["total_packs_sold"] = df["total_bottles_sold"] / prepared_sample["pack_size"]
    df["day"] = df['date'].dt.day
    prepared_sample["last_valid_day"] = int(df["day"].iloc[-1])
    df.set_index("day", drop=False, inplace=True)
    # ensure all samples are same sequence length
    df = df.reindex(range(1, SEQ_LEN + 1))
    # don't use padded days
    df["valid_day"] = df["valid_day"].fillna(False)
    # 0 index
    df["day"] = df.index - 1
    df = df.ffill().bfill()
    for feature in ONE_DIM_INT_FEATURES:
        prepared_sample[feature] = df[feature].values.astype(int)
    for feature in ONE_DIM_FLOAT_FEATURES:
        prepared_sample[feature] = df[feature].values.astype(float)
    validate_sample(prepared_sample)
    return prepared_sample


def validate_sample(prepared_sample):
    """
    Check that the sample matches the schema.

    Args:
        prepared_sample (dict): should match specified data schema

    """
    sample_id = '_'.join([str(prepared_sample[k]) for k in ['item_id', 'year', 'month']])
    # final check on shapes and types to make sure that TFRecord can be made
    for k in DATA_FEATURE_SPEC:
        v = prepared_sample.get(k)
        if v is None:
            err_msg = "{} is missing {}".format(sample_id, k)
            logging.error(err_msg)
            raise AssertionError(err_msg)
        if isinstance(v, np.ndarray):
            logging.debug('%s shape: %s', k, v.shape)
            try:
                assert v.shape == tuple(DATA_FEATURE_SPEC[k].shape)
            except AssertionError:
                err_msg = "{}: {} shape {} but should be {}".format(sample_id,
                                                                    k, v.shape,
                                                                    tuple(DATA_FEATURE_SPEC[k].shape))
                logging.error(err_msg)
                raise AssertionError(err_msg)
        else:
            if k in SCALAR_STRING_FEATURES:
                try:
                    assert isinstance(v, str)
                except AssertionError:
                    err_msg = "{}: {} is {} but should be string".format(sample_id, k, v)
                    logging.error(err_msg)
                    raise AssertionError(err_msg)
            else:
                try:
                    assert isinstance(v, (int, float))
                except AssertionError:
                    err_msg = "{}: {} is {} but should be numeric".format(sample_id, k, v)
                    logging.error(err_msg)
                    raise AssertionError(err_msg)


def preprocessing_fn(input_tensors):
    """
    Preprocessing for full-pass transformations (e.g. scaling).
    tf.Transform operations are done with TensorFlow model itself.

    Args:
        input_tensors (dict): dict of tensors that came from prepare_seq_sample

    Returns:
        dict: preprocessed feature and label tensors

    """
    output_tensors = input_tensors.copy()
    output_tensors["volume_ml"] = tft.scale_to_0_1(input_tensors["volume_ml"])
    output_tensors["pack_size"] = tft.scale_to_0_1(input_tensors["pack_size"])
    return output_tensors


def split_train_test(sample, num_partitions, split_dt):
    """
    Split dataset into train and test.

    Args:
        sample (dict): departure_date should exist as key
                       value is in bytes
        num_partitions (int): number of partitions
        split_dt (datetime): samples with departure date >= this will be
                             in test set, rest in train

    Returns:
        int: 0 for train, 1 for test

    """
    # rescale up 0-indexed month for splitting
    return int(datetime(sample["year"], sample["month"] + 1, 1) >= split_dt)


class Predict(beam.DoFn):
    def __init__(self,
                 model_dir,
                 meta_tag=tf.saved_model.SERVING,
                 meta_signature=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
        self.model_dir = model_dir
        self.meta_tag = meta_tag
        self.meta_signature = meta_signature
        self.session = None
        self.graph = None
        self.feed_tensors = None
        self.fetch_tensors = None

        # Beam metrics
        self.__name__ = "Predict: {}".format(model_dir)
        self._elements_counter = Metrics.counter(self.__name__, 'n_elements')
        # Create a counter to measure number of bad elements.
        self._bad_elements_counter = Metrics.counter(self.__name__, 'failed_elements')
        # major python builtin exceptions
        self._assert_err = Metrics.counter(self.__name__, 'AssertionErrors')
        self._key_err = Metrics.counter(self.__name__, 'KeyErrors')
        self._type_err = Metrics.counter(self.__name__, 'TypeErrors')
        self._val_err = Metrics.counter(self.__name__, 'ValueErrors')
        self._other_err = Metrics.counter(self.__name__, 'other_errors')

    def process(self, inputs):
        """
        Args:
            inputs (dict): from prepare_sample

        Returns:
            dict: predicted outputs - should match BigQuery schema if results
                  will be written to table

        """
        try:
            # Create a session for every worker only once. The session is not
            # pickleable, so it can't be created at the DoFn constructor.
            if not self.session:
                self.graph = ops.Graph()
                with self.graph.as_default():
                    self.session = tf.compat.v1.Session()
                    # load(sess, tags, export_dir, import_scope=None, **saver_kwargs)
                    metagraph_def = tf.compat.v1.saved_model.load(self.session,
                                                                  {self.meta_tag},
                                                                  self.model_dir)
                signature_def = metagraph_def.signature_def[self.meta_signature]
                assert tf.compat.v1.saved_model.is_valid_signature(signature_def)
                # inputs
                self.feed_tensors = {
                    k: self.graph.get_tensor_by_name(v.name)
                    for k, v in signature_def.inputs.items()
                }
                # outputs/predictions
                self.fetch_tensors = {
                    k: self.graph.get_tensor_by_name(v.name)
                    for k, v in signature_def.outputs.items()
                }
            # Create a feed_dict for a single element.
            feed_dict = {
                tensor: [inputs[key]]
                for key, tensor in self.feed_tensors.items() if key in inputs
            }
            results = self.session.run(self.fetch_tensors, feed_dict)
            results = next(iter(results.values()))
            yield {
                'item_number': inputs['item_id'],
                'pred_date': '2020-07-16',
                'pred_total_pack': results[0][inputs['last_valid_day']][0]
            }

        except Exception as e:  # pylint: disable=broad-except
            logging.error("%s failed for %s due to unhandled exception %s",
                          self.__name__, inputs, e, exc_info=True)
            # Catch any exception the above call.
            self._bad_elements_counter.inc(1)
            if isinstance(e, AssertionError):
                self._assert_err.inc(1)
            elif isinstance(e, KeyError):
                self._key_err.inc(1)
            elif isinstance(e, TypeError):
                self._type_err.inc(1)
            elif isinstance(e, ValueError):
                self._val_err.inc(1)
            else:
                self._other_err.inc(1)
        finally:
            self._elements_counter.inc(1)
