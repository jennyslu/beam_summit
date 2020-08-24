import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

# sequence length
SEQ_LEN = 31
SCALAR_STRING_FEATURES = [
    "item_description",
]
SCALAR_INT_FEATURES = [
    "item_id",
    "year",
    "month",
    "last_valid_day"
]
SCALAR_FLOAT_FEATURES = [
    "volume_ml",
    "pack_size",
    "cost",
    "retail"
]
ONE_DIM_INT_FEATURES = [
    "day",
    "valid_day",
]
ONE_DIM_FLOAT_FEATURES = [
    "cumu_bottles_sold",
    "daily_bottles_sold",
    "total_packs_sold"
]
TARGET = "total_packs_sold"
# spec for TFRecords
DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.string))
     for name in SCALAR_STRING_FEATURES] +
    [(name, tf.io.FixedLenFeature([], tf.int64))
     for name in SCALAR_INT_FEATURES] +
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in SCALAR_FLOAT_FEATURES] +
    [(name, tf.io.FixedLenFeature([SEQ_LEN], tf.int64))
     for name in ONE_DIM_INT_FEATURES] +
    [(name, tf.io.FixedLenFeature([SEQ_LEN], tf.float32))
     for name in ONE_DIM_FLOAT_FEATURES]
)
# contains schema and defins layout of data
DATA_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(DATA_FEATURE_SPEC))
TRAIN_FILES_PATTERN = "train"
EVAL_FILES_PATTERN = "test"
