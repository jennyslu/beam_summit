import argparse
import json
import logging
import os
import time
from datetime import datetime

import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions

import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from iowa_sales.beam import (BQQuery, MapAndFilterErrors, Predict,
                             prepare_seq_sample, preprocessing_fn,
                             split_train_test)
from iowa_sales.config import (DATA_METADATA, EVAL_FILES_PATTERN,
                               TRAIN_FILES_PATTERN)
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Iowa liquor sales pipeline")
    parser.add_argument("--data_dir", type=str, default=os.path.join(CURRENT_DIR, 'iowa_sales/data'),
                        help='data directory (either local or on GCS)')
    parser.add_argument("--run_cloud", action="store_true",
                        help="run in cloud instead of locally. default false")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"],
                        help="train model or use saved model to make predictions")
    parser.add_argument("--output_dir", type=str, default=datetime.now().strftime("%Y%m%d%H%M"),
                        help='name of output folder within datadir/tfrecords. must be specified if mode is train.')
    parser.add_argument("--model_dir", type=str, default='202008240050/1598255430',
                        help='location of saved model within datadir/model/. must be specified if mode is not train.')
    args = parser.parse_args()

    if args.run_cloud:
        logging.info('running in cloud on DataFlow')
        # see https://cloud.google.com/dataflow/docs/guides/specifying-exec-params for more details
        argv = [
            '--runner', 'DataflowRunner',
            # '--project', PROJECT,
            '--staging_location', os.path.join(args.data_dir, "staging"),
            '--temp_location', os.path.join(args.data_dir, "temp"),
            # see https://cloud.google.com/dataflow/docs/concepts/regional-endpoints for more details
            # '--region', REGION,
            # see https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline#dataflow-shuffle
            # '--experiments', 'shuffle_mode=service',
            # see https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/ for more details
            '--setup_file', os.path.join(CURRENT_DIR, 'setup.py'),
        ]
        options = PipelineOptions(flags=argv)
    else:
        logging.info('running locally on DirectRunner')
        argv = [
            '--runner', 'DirectRunner',
            '--staging_location', os.path.join(args.data_dir, "staging"),
            '--temp_location', os.path.join(args.data_dir, "temp"),
            '--setup_file', os.path.join(CURRENT_DIR, 'setup.py'),
        ]
        options = PipelineOptions(flags=argv)

    t1 = time.time()
    with tft_beam.Context(temp_dir=options.display_data()['temp_location']):
        pipeline = beam.Pipeline(options=options)
        # when training we want all the data
        if args.mode == "train":
            logging.info("TRAINING")
            if args.run_cloud:
                source = (
                    pipeline
                    | 'Read BQ table' >> beam.io.Read(
                        beam.io.gcp.bigquery.BigQuerySource(
                            query=BQQuery.train_query, use_standard_sql=True))
                )
            else:
                source = (
                    pipeline
                    | 'Read local JSON' >> beam.io.ReadFromText(
                        os.path.join(args.data_dir, 'bq_sample.json'))
                    | 'Parse JSON' >> MapAndFilterErrors(json.loads)
                )
        # when doing predictions, we want to get the latest month of data
        elif args.mode == "predict":
            logging.info("PREDICTIONS")
            if args.run_cloud:
                source = (
                    pipeline
                    | 'Read BQ table' >> beam.io.Read(
                        beam.io.gcp.bigquery.BigQuerySource(
                            query=BQQuery.predict_query, use_standard_sql=True))
                )
            else:
                source = (
                    pipeline
                    | 'Read local JSON' >> beam.io.ReadFromText(
                        os.path.join(args.data_dir, 'bq_sample_predict.json'))
                    | 'Parse JSON' >> MapAndFilterErrors(json.loads)
                )

        # create sequence samples
        prepared_samples = (
            source
            | 'Prepare samples' >> MapAndFilterErrors(prepare_seq_sample)
        )

        if args.mode == "train":
            # TFTransform preprocessing done over entire dataset
            (transformed_data_and_metadata, transform_fn) = (
                (prepared_samples, DATA_METADATA)
                | 'TFT AnalyzeAndTransformDataset feature scaling' >>
                tft.beam.AnalyzeAndTransformDataset(preprocessing_fn)
                )
            transformed_data, transformed_metadata = transformed_data_and_metadata
            # split into train and test
            train, test = (
                transformed_data
                | 'Partition train/test' >> beam.Partition(
                    split_train_test, 2, split_dt=datetime(2020, 1, 1)
                )
            )
            # encoder for TFRecords
            transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)
            # write train dataset
            _ = (
                train
                | 'Encode & write train -> TFRecords' >> tfrecordio.WriteToTFRecord(
                    file_path_prefix=os.path.join(args.data_dir, 'tfrecords', args.output_dir, TRAIN_FILES_PATTERN),
                    coder=transformed_data_coder,
                    file_name_suffix='.gz',
                    num_shards=4,
                    compression_type=beam.io.filesystem.CompressionTypes.GZIP
                    ))
            # write validation dataset
            _ = (
                test
                | 'Encode & write test -> TFRecords' >> tfrecordio.WriteToTFRecord(
                    file_path_prefix=os.path.join(args.data_dir, 'tfrecords', args.output_dir, EVAL_FILES_PATTERN),
                    coder=transformed_data_coder,
                    file_name_suffix='.gz',
                    num_shards=1,
                    compression_type=beam.io.filesystem.CompressionTypes.GZIP
                    ))
            # write the transform_fn
            _ = (
                transform_fn
                | 'Write transformFn' >> transform_fn_io.WriteTransformFn(
                    os.path.join(args.data_dir, 'tfrecords', args.output_dir)
                    ))
        else:
            predictions = (
                prepared_samples
                | 'Predict' >> beam.ParDo(Predict(
                    model_dir=os.path.join(args.data_dir, 'models', args.model_dir)
                    ))
            )
            _ = predictions | 'Print predictions' >> beam.Map(print)
            '''
            _ = (
                predictions
                | 'Write to BQ' >> beam.io.WriteToBigQuery(
                    table=PRED_TABLE,
                    schema={
                        'fields': [
                            {'name': 'item_number', 'type': 'INTEGER', 'mode': 'REQUIRED'},
                            {'name': 'pred_date', 'type': 'DATE', 'mode': 'REQUIRED'},
                            {'name': 'pred_total_packs', 'type': 'FLOAT', 'mode': 'REQUIRED'}
                        ]
                    },
                    dataset=PRED_DATASET,
                    project=PROJECT,
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))
            '''

        # wait for pipeline result to finish
        pipeline.run().wait_until_finish()
        if args.mode == "train":
            logging.info("wrote TFRecords to %s", os.path.join(args.data_dir, 'tfrecords', args.output_dir))
            logging.info("pipeline took %.2f", time.time() - t1)
