# Slides
Slides can be found [here](https://docs.google.com/presentation/d/1na_3xv9eglSvKGDj_zhZxPa5CEfmQ8-rMdX1MJPDdR4/edit?usp=sharing)

# Local setup
Install requirements as per `setup.py`

# GCP setup
When you first Google Cloud account, there will be a default project called `Quickstart` you can use. You will have to add some billing information.

# Using Beam to generate dataset (TFRecords)
There is a small sample of the query result saved within `/data`. For running locally, this will be read in but for running on cloud query will be executed. The result of both is equivalent: a dict representing each row.

## Beam pipeline options

`--runner` determines the backend. Two options are demonstrated here, although there are more: `DirectRunner`, which will run locally and `DataFlow`, which is part of Google Cloud Platform

`--staging_location` and `--temp_location` can be GCS buckets or local directories. Temporary files 

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

`beam.py`: lines 138-228
Here we format our sequence samples and also do some very simple feature engineering.
```python
# sample from input to test with
raw_sample = {'item_number': '36904', 'item_description': 'Mccormick Vodka Pet', 'cost': 1.76, 'retail': 2.63, 'volume_ml': 375, 'pack_size': 24, 'year_month': '2012-01', 'dates_arr': ['2012-01-03', '2012-01-04', '2012-01-05', '2012-01-09', '2012-01-10', '2012-01-11', '2012-01-12', '2012-01-16', '2012-01-17', '2012-01-18', '2012-01-19', '2012-01-23', '2012-01-24', '2012-01-25', '2012-01-26', '2012-01-30', '2012-01-31'], 'daily_bottles_sold_arr': ['570', '303', '922', '570', '573', '433', '889', '673', '923', '537', '1077', '735', '754', '554', '969', '639', '630']}
```

### 3. Input scaling/standardization/normalization across dataset
`main.py`: lines 112-118

`beam.py`: lines 196-211
Here we apply some simplistic scalings on some of the inputs. These transformations can involve doing a full pass over the entire dataset to compute an aggregate statistic such as min or max to scale.

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
tfrecord_pattern = 'iowa_sales/data/tfrecords/202008232054/train*.gz'
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

We will also want to apply some preprocessing to reshape the tensors to the shapes we want for our model. We also have to separate the features from the labels.
```python

```

Finally we want to batch the dataset for training and evaluation.
```python
batched_dataset = parsed_dataset.batch(10)
for x in batched_dataset.take(1):
    # this "x" is a useful sample to debug and test model code with
    print(repr(x))
```

## Train RNN
```python
# train and export a model using previously generated dataset
%run iowa_sales/model.py '202008232251'
```

# Using Beam to make predictions

## Prediction pipeline transformations
Having a unified training and prediction pipeline is crucial - if features were being processed differently between the two avenues then results would not be valid. Using Beam we can easily add prediction abilities to the same pipeline we just used to generate training datasets. Our example scenario is more suited to batch style inference but Beam can also perform streaming predictions.

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

### 5. Save results to BigQuery
Not demonstrated here since not all participants are assumed to have GCP set up but sample code for how to write results to BigQuery can be seen.
