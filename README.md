# Slides
Slides can be found [here](https://docs.google.com/presentation/d/1FyuAlalH8yB6kLJCPokInPhjW4gRuWy35fuJFpTkl5I/edit?usp=sharing)

# Local setup
Install requirements as per `setup.py`

# GCP setup
When you first Google Cloud account, there will be a default project called `Quickstart` you can use. You will have to add some billing information.

# Data
We are using a small public dataset available in BigQuery: `bigquery-public-data.iowa_liquor_sales.sales`.
If we look at the table we can see it is transactional data on liquor sales, with information about the store, vendor and the sale itself. We are primarily interested in the columns about which item was sold, how much, and details about the item itself. 

# Using Beam to generate dataset (TFRecords)
There is a small sample of the query result saved within `/data`. For running locally, this will be read in but for running on cloud query will be executed. The result of both is equivalent: a dict representing each row.

## Beam pipeline options

`--runner` determines the backend. Two options are demonstrated here, although there are more: `DirectRunner`, which will run locally and `DataFlow`, which is part of Google Cloud Platform

`--staging_location` and `--temp_location` can be GCS buckets or local directories. Temporary files will be written here if necessary.

`--setup_file` is required when you use external packages (as we are in this example) for when you want to run on a cloud backend so that the workers can install the correct setup.

### DataFlow specific options
There are many options, which you can read more about in DataFlow docs, but some important ones to know are `--project`, `--region`, and `--experiments`. A useful "experiment" is `shuffle_mode`, which is able to **dramatically** speed up aggregations. However, this service is only available in certain regions, and regions are also limited in the type of workers you can get. Be sure to check what is available in your region.

## Training pipeline transformations

### 1. Input
`main.py`: lines 69-103
Debugging can be a bit tricky. Some tips I have found:
  * work with a smaller local sample of your data. Here's I've exported a small sample of the query I would ultimately want to use on the entire dataset. We can read in newline-delimited JSON and get the same result as if we were reading from BigQuery directly. 
  * debuggers like `pdb` won't really work in the pipeline. Instead, one can insert a `beam.Map(print)` or some other simple function to see what you have in any given PCollection

### 2. Data preprocessing
`main.py`: lines 105-109

`beam.py`: lines 87-135
Here we have defined a convenient wrapper for a DoFn. When PTransforms are applied the `expand` method is called and passed input PCollection. When DoFns are applied the `process` method is called and passed a single record from an input PCollection. DoFns could return 0 or more outputs for 1 given input. Here you can see the allowed format for output is either a list or a single non-list object. In either case the results should be yielded.
If any step in your pipeline fails too many times your entire pipeline will fail. You may want to keep this behaviour but most likely you just want to skip these bad samples for now and log the issue so you can re-visit it later. Beam `Metrics` allow you can use to keep track of import values, such as number of records, number of failures, etc. I've implement a simple version here that keeps track of a few different assertion errors we might see later on but you could obviously expand this.

`config.py`
Our feature specifications are defined here. You can see we've defined some scalar features as well as some sequence features (the one dimensional ones).

`beam.py`: lines 138-228
Here we format our sequence samples and also do some very simple feature engineering.
```python
# sample from input to test with
raw_sample = {'item_number': '36904', 'item_description': 'Mccormick Vodka Pet', 'cost': 1.76, 'retail': 2.63, 'volume_ml': 375, 'pack_size': 24, 'year_month': '2012-01', 'dates_arr': ['2012-01-03', '2012-01-04', '2012-01-05', '2012-01-09', '2012-01-10', '2012-01-11', '2012-01-12', '2012-01-16', '2012-01-17', '2012-01-18', '2012-01-19', '2012-01-23', '2012-01-24', '2012-01-25', '2012-01-26', '2012-01-30', '2012-01-31'], 'daily_bottles_sold_arr': ['570', '303', '922', '570', '573', '433', '889', '673', '923', '537', '1077', '735', '754', '554', '969', '639', '630']}
```

### 3. Input scaling/standardization/normalization across dataset
The preprocessing function is the most important conecpt of Tensorflow Transform. It describes a transformation of the dataset. Two types of functions are used to define preprocessing functions:
  1. function that accepts and returns tensors
  2. analyzers that compute a full-pass over dataset to generate constant value (e.g. mean, min, max) that can be used

TF Transform provides canonical implementation on Beam with two primary `PTransforms`: `AnalyzeDataset` and `TransformDataset`, as well as the combined `AnalyzeAndTransformDataset`, which is easier to use.

`main.py`: lines 112-118

`beam.py`: lines 196-211
Here we apply some simplistic scalings on some of the inputs to 0 and 1. Obviously this can be much more complex but here we demonstrate how to integrate TFT into our Beam pipeline.

### 4. Split into train and validate
`main.py`: lines 119-125

`beam.py`: lines 214-230
Generally speaking lambda functions that require external arguments will fail in Beam pipelines so even though this is a very simple function we define it here. Partition functions take in the element, number of partitions, and potential additional arguments. Here because we are dealing with time series data, we use a datetime as a partitioning field.

### 5. Encode and write to TFRecords
`main.py`: lines 126-153

```python
# running locally
%run main.py --mode 'train'
```

# Build RNN using above dataset
It's always helpful to have a "sample" batch to debug while you build model. We will only build a very simple one here. 

## Examine TFRecords
We can take a look at what the TFRecords look like.
```python
tfrecord_pattern = 'iowa_sales/data/tfrecords/202008232251/train*.gz'
files = tf.data.Dataset.list_files(tfrecord_pattern)
raw_dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
for raw_record in raw_dataset.take(3):
    print(repr(raw_record))
```

To create our dataset, we will want to parse the records.
```python
def _parse_tfrecord(elem):
    return tf.io.parse_single_example(elem, features=DATA_FEATURE_SPEC)
parsed_dataset = raw_dataset.map(_parse_tfrecord)
for parsed_record in parsed_dataset.take(3):
    print(repr(parsed_record))
```

We have to separate features from target; the expected output of model input function is a tuple of two dicts, 1st element being the features and second being target.
```python
def _pop_target(features):
    labels = features.pop(TARGET, {})
    return features, labels
split_dataset = parsed_dataset.map(_pop_target)
for split_record in split_dataset.take(3):
    print(repr(split_record))
```

We also want to batch the dataset for training and evaluation.
```python
batched_dataset = parsed_dataset.batch(10)
for batched_record in batched_dataset.take(1):
    # this "x" is a useful sample to debug and test model code with
    print(repr(batched_record))
```

Now we need to apply some preprocessing to manipulate the shapes of tensors into what we want for our model. This part can be a bit tricky as this function needs to work for batches as well as single records and the batch size for the serving function is `None`.
Here I've done a very simple expansion of the scalar features so that they are repeated for all the time steps as well as dimension expansion on all features so that the input ultimately matches the expected `(batch, time, feature)` shape.
```python
# single example
features = parsed_record
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

# batched example
features = batched_record
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
```

## Train RNN
```python
# train and export a model using previously generated dataset
%run iowa_sales/model.py '202008232251'
```
When debugging model, it is often helpful to have a sample of the data to test with. As of TF 2.0 eager execution is default so debugging has become much easier. Generally mismatches tensor shapes are the most common errors, which is why it's very useful to test with a sample and check your shapes at every stage.

### Building model
Here we use two embeddings as a lookup for 2 categorical features: `item_id` and `month`. Embedding lookups add an extra dimension to the input tensor, so they need to be reshaped in order to be concatenated with the other inputs. The other inputs are simple numeric sequences that we concatenate to the embedding vectors.

All of this is put into a simple LSTM with a dense layer with 1 output for each day. We return sequences here because we are interested in the prediction every day - not only the final day.
```python
month = keras.Input(shape=(SEQ_LEN, 1), name="month")
month_lookup = layers.Embedding(12, 2, name="month_lookup")
# shape: (batch, seq, 1, embedding) -> (batch, seq, embedding)
print(month_lookup(month), (-1, SEQ_LEN, 2))
print(tf.reshape(month_lookup(month), (-1, SEQ_LEN, 2)))

# regular sequence feature
daily = keras.Input(shape=(SEQ_LEN, 1), name="daily_bottles_sold")
# shape: (batch, time, features)
inputs = layers.concatenate([
    tf.reshape(month_lookup(month), (-1, SEQ_LEN, 2)),
    daily
    ])
print(inputs.shape)

# simple model
lstm_model = tf.keras.models.Sequential([
    # shape: (batch, time, features) => (batch, time, lstm_units)
    tf.keras.layers.LSTM(32, return_sequences=True),
    # shape: => (batch, time, outputs)
    tf.keras.layers.Dense(units=1)
    ])
outputs = lstm_model(inputs)
print(outputs.shape)
```

### Exporting model
A serving input function needs to be provided in order to save and export model. This effectively defines the "API" of the model. The input serving function uses feature placeholders - **which should match the feature spec we used in our Beam pipeline**. Batch size should be `None` and this means that some of the preprocessing can be tricky.

# Using Beam to make predictions
Now that we have a trained model saved and exported, we can use it to serve predictions using the same Beam pipeline.

## Prediction pipeline transformations
Having a unified training and prediction pipeline is crucial - if features were being processed differently between the two avenues then results would not be valid. Using Beam we can easily add prediction abilities to the same pipeline we just used to generate training datasets. Our example scenario is more suited to batch style inference but Beam also has the ability to use streaming data.

## Test prediction with SavedModel

### 1. Input
Here I've pretended that we are "on" a certain make making a prediction but in reality you would simply use the same query as training with filters for date ranges that would automatically only exist up until "now".

### 2. Data preprocessing
Exactly the same function as above.

### 3. Input scaling/standardization/normalization across dataset
Exactly the same function as above.

### 4. Load SavedModel and make predictions
Create a single sample to test predictions.
```python
raw_sample = {'item_number': '68039', 'item_description': 'Baileys Original Irish Cream 100ml', 'cost': 41.76, 'retail': 62.64, 'volume_ml': 2400, 'pack_size': 4, 'year_month': '2020-07', 'dates_arr': ['2020-07-01', '2020-07-02', '2020-07-03', '2020-07-06', '2020-07-07', '2020-07-08', '2020-07-09', '2020-07-13', '2020-07-15'], 'daily_bottles_sold_arr': ['5', '2', '10', '8', '4', '8', '16', '6', '12']}
prepared_sample = prepare_seq_sample(raw_sample)
```

Test making predictions with your saved model.
```python
model_dir = 'iowa_sales/data/models/202008240050/1598255430'
session = None
if not session:
    graph = ops.Graph()
    with graph.as_default():
        session = tf.compat.v1.Session()
        # load(sess, tags, export_dir, import_scope=None, **saver_kwargs)
        metagraph_def = tf.compat.v1.saved_model.load(session,
                                                      {tf.saved_model.SERVING},
                                                      model_dir)
    signature_def = metagraph_def.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    assert tf.compat.v1.saved_model.is_valid_signature(signature_def)
    # inputs
    feed_tensors = {
        k: graph.get_tensor_by_name(v.name)
        for k, v in signature_def.inputs.items()
    }
    # outputs/predictions
    fetch_tensors = {
        k: graph.get_tensor_by_name(v.name)
        for k, v in signature_def.outputs.items()
    }
# Create a feed_dict for a single element.
feed_dict = {
    tensor: [prepared_sample[key]]
    for key, tensor in feed_tensors.items() if key in prepared_sample
}
results = session.run(fetch_tensors, feed_dict)
```

Because we have a sequence model that is returning a prediction for every day, we need to chose the correct index. This is where `last_valid_day` comes in. Although this is not completely necessary in our setup, if we were predicting T+1 at every T, then the sequence indexing would be more crucial and having a "marker" in the data of which "current" index to us is quite helpful.

### 5. Save results to BigQuery
Not demonstrated here since not all participants are assumed to have GCP set up but sample code for how to write results to BigQuery can be seen.
