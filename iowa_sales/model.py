import argparse
import os
from datetime import datetime
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_transform as tft
from iowa_sales.config import (DATA_FEATURE_SPEC, EVAL_FILES_PATTERN, SEQ_LEN,
                               TARGET, TRAIN_FILES_PATTERN)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MAX_EPOCHS = 20


def _preprocess(features, mode):
    """
    Shape all features to be what we want, i.e. (batch, time, feature)

    Args:
        features (dict): of tensors
        mode (str): one of tf.estimators.ModeKeys

    Returns:
        dict: with keys of features that are used in model

    """
    output_features = {"valid_day": features["valid_day"]}
    # (time, 1)
    for f in ["item_id", "month", "cost", "retail", "pack_size"]:
        if features[f].shape == ():
            output_features[f] = tf.expand_dims(tf.tile(tf.expand_dims(features[f], 0), multiples=[SEQ_LEN]), 1)
            rank = 1
        # ranks don't match - could also check mode here
        else:
            # recommend that you use multiply instead of alternatives
            # e.g. reshape, tile, etc. as those will often not work for serving
            output_features[f] = tf.expand_dims(
                tf.multiply(tf.expand_dims(features[f], 1),
                            tf.ones(SEQ_LEN, dtype=features[f].dtype)), 2)
            rank = 2
    for f in ["cumu_bottles_sold", "daily_bottles_sold", "total_packs_sold"]:
        if rank == 1:
            output_features[f] = tf.expand_dims(features[f], 1)
        elif rank == 2:
            output_features[f] = tf.expand_dims(features[f], 2)
    return output_features


def make_input_fn(tfrecord_pattern,
                  feature_spec,
                  target,
                  batch_size,
                  mode):
    def _parse_tfrecord(elem):
        return tf.io.parse_single_example(elem, features=feature_spec)

    def _pop_target(features):
        labels = features.pop(target, {})
        return features, labels

    def input_fn():
        """
        Estimator `input_fn`.

        Returns:
            tuple: 1st element is dict mapping features: Tensor`
                   2nd element is `Tensor` of target

        """
        files = tf.data.Dataset.list_files(tfrecord_pattern)
        # process 4 files concurrently and interleave records from each file
        dataset = files.interleave(
            lambda files: tf.data.TFRecordDataset(files, compression_type="GZIP"),
            cycle_length=4,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # parse tfrecords
        dataset = dataset.map(_parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        preprocess = partial(_preprocess, mode=mode)
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # split features and label
        dataset = dataset.map(_pop_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # cache dataset for training only
            dataset = dataset.cache()
            # shuffle dataset for training
            dataset = dataset.shuffle(buffer_size=20000)
            dataset.repeat()
        # combine consecutive elements into batches
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    return input_fn


def make_serving_input_fn(tft_output, input_feature_spec, target):
    """
    Make input_fn for serving predictions.
    Apply the transformations from TFT preprocessing_fn before serving.
    """
    def serving_input_fn():
        feature_placeholders = {}
        for feature_name in input_feature_spec:
            feature_placeholders[feature_name] = tf.compat.v1.placeholder(
                input_feature_spec[feature_name].dtype,
                shape=[None] + input_feature_spec[feature_name].shape,
                name=feature_name)
        inputs = tft_output.transform_raw_features(feature_placeholders)
        preprocess = partial(_preprocess, mode=tf.estimator.ModeKeys.EVAL)
        inputs = preprocess(inputs)
        del feature_placeholders[target]
        del inputs[target]
        return tf.estimator.export.ServingInputReceiver(inputs, feature_placeholders)

    return serving_input_fn


def make_simple_rnn():
    """
    Create simple RNN.
    """
    # features to be tiled
    item_id = keras.Input(shape=(SEQ_LEN, 1), name="item_id")
    month = keras.Input(shape=(SEQ_LEN, 1), name="month")
    item_lookup = layers.Embedding(999450, 10, name="item_lookup")
    month_lookup = layers.Embedding(12, 2, name="month_lookup")
    cost = keras.Input(shape=(SEQ_LEN, 1), name="cost")
    retail = keras.Input(shape=(SEQ_LEN, 1), name="retail")
    pack = keras.Input(shape=(SEQ_LEN, 1), name="pack_size")
    cumu = keras.Input(shape=(SEQ_LEN, 1), name="cumu_bottles_sold")
    daily = keras.Input(shape=(SEQ_LEN, 1), name="daily_bottles_sold")
    mask = layers.Input(shape=(SEQ_LEN,), name="valid_day")
    # shape: (batch, time, features)
    inputs = layers.concatenate([
        tf.reshape(item_lookup(item_id), (-1, SEQ_LEN, 10)),
        tf.reshape(month_lookup(month), (-1, SEQ_LEN, 2)),
        cost, retail, pack, cumu, daily
        ])
    lstm_model = tf.keras.models.Sequential([
        # shape: (batch, time, features) => (batch, time, lstm_units)
        tf.keras.layers.LSTM(32, return_sequences=True),
        # shape: => (batch, time, outputs)
        tf.keras.layers.Dense(units=1)
        ])
    outputs = lstm_model(inputs, mask)
    model = keras.Model(inputs=[item_id, month, cost, retail, pack, cumu, daily, mask],
                        outputs=outputs)
    return model


def train_and_evaluate(model_dir,
                       input_feature_spec,
                       target,
                       train_files_pattern,
                       eval_files_pattern,
                       batch_size=64,
                       train_max_steps=1000):
    """
    Trains and evaluates the estimator given.
    The input functions are generated by the preprocessing function.
    """
    # specify where model is stored
    if tf.io.gfile.exists(model_dir):
        tf.io.gfile.rmtree(model_dir)
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=model_dir)
    # this will give us a more granular visualization of the training
    run_config = run_config.replace(save_summary_steps=1)

    # no build in RNN estimator in TF yet
    model = make_simple_rnn()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model
    )

    # wrapper around output of tf.Transform
    tft_output = tft.TFTransformOutput(os.path.split(train_files_pattern)[0])
    feature_spec = tft_output.transformed_feature_spec()

    # Create the training and evaluation specifications
    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(
            tfrecord_pattern=train_files_pattern,
            feature_spec=feature_spec,
            target=target,
            batch_size=batch_size,
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=train_max_steps
        )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(
            tfrecord_pattern=eval_files_pattern,
            feature_spec=feature_spec,
            target=target,
            batch_size=batch_size,
            mode=tf.estimator.ModeKeys.EVAL))
    # train and evaluate the model
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # export saved model
    estimator.export_saved_model(
        model_dir,
        serving_input_receiver_fn=make_serving_input_fn(
            tft_output, input_feature_spec, target
            ))


if __name__ == '__main__':
    """Main function called by AI Platform."""
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('dataset_dir', type=str,
                        help='dataset location within working_dir/tfrecords/<dataset_dir>')
    parser.add_argument('--working_dir', type=str,
                        default=os.path.join(CURR_DIR, 'data'),
                        help='working directory (could be local or on GCS)')
    parser.add_argument('--model_dir', type=str, default=datetime.now().strftime("%Y%m%d%H%M"),
                        help='dir to save model within working_dir/models/<model_dir>')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training and evaluation.')
    parser.add_argument('--train_max_steps', type=int, default=100,
                        help='number of steps to train the model')
    args = parser.parse_args()

    train_and_evaluate(model_dir=os.path.join(args.working_dir, 'models', args.model_dir),
                       input_feature_spec=DATA_FEATURE_SPEC,
                       target=TARGET,
                       train_files_pattern=os.path.join(
                           args.working_dir, 'tfrecords', args.dataset_dir,
                           "{}*".format(TRAIN_FILES_PATTERN)),
                       eval_files_pattern=os.path.join(
                           args.working_dir, 'tfrecords', args.dataset_dir,
                           "{}*".format(EVAL_FILES_PATTERN)),
                       batch_size=args.batch_size,
                       train_max_steps=args.train_max_steps)
