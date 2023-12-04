import os
import shutil
import csv
import tensorflow as tf
import polars as pl
import seaborn as sns
import numpy as np

from os.path import isfile
from functools import lru_cache
from typing import Dict, List
from loguru import logger
from os.path import join

from sqlalchemy import create_engine

VOCAB = ['A','C','G','U']
MAX_LEN = 457


def string_lookup():
    return tf.keras.layers.StringLookup( vocabulary=VOCAB,mask_token=None,output_mode="int")


def training_strategy():
    tpu = None
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local") # "local" for 1VM TPU
        strategy = tf.distribute.TPUStrategy(tpu)
        return strategy
    except:
        strategy = tf.distribute.get_strategy()
        return strategy
    


def get_train_parquet_path(no_nulls:bool):
    if no_nulls is True:
        return "/home/kineticengines/app/datasets/train_data_no_nulls.parquet"
    else:
        return "/home/kineticengines/app/datasets/train_data_with_nulls.parquet" 
    

def get_train_tfrecords_path(no_nulls:bool):
    if no_nulls is True:
        return "/home/kineticengines/app/datasets/as_tfrecords/pl2tf/dataset_no_nulls/"
    else:
        return "/home/kineticengines/app/datasets/as_tfrecords/pl2tf/dataset_with_nulls/" 


def load_training_dataset(no_nulls:bool):   
    train_dataset_as_parquet =  get_train_parquet_path(no_nulls)
    if isfile(train_dataset_as_parquet):
        logger.info("Reading from existing parquet dataset")
        return pl.scan_parquet(train_dataset_as_parquet)      
    else:
        logger.info("Creating new parquet dataset")
        train_data_path = "/home/kineticengines/app/datasets/train_data_QUICK_START.csv"
        df = pl.scan_csv(train_data_path)
        schema = df.schema
        del df

        for key,_ in schema.items():
            if key == "sequence_id" or key == "dataset_name" or \
            key == "reads" or key == "signal_to_noise" or key == "SN_filter" or \
            key == 'sequence' or key == 'experiment_type':
                continue
            else:
                schema[key] = pl.Float64  

        df = pl.scan_csv(train_data_path,schema=schema).drop("sequence_id","dataset_name",
                "reads","signal_to_noise","SN_filter")

        if no_nulls is True:
            df = df.fill_null(0.00) 

        df.sink_parquet(train_dataset_as_parquet,compression_level=20) 
        del df
        return pl.scan_parquet(train_dataset_as_parquet)  


@lru_cache
def load_testdataset():
    logger.info("Reading test dataset")
    test_data_path = "/home/kineticengines/app/datasets/test_sequences.csv"
    df = pl.scan_csv(test_data_path).drop("sequence_id","future").collect(streaming=True)    
    return df


def check_experiments_profile_count():
    """
    Check the distribution on experiments. Expectation is that the two(2) experiments 
    have the same distribution
    """
    df = load_training_dataset()   
    df = df.select(["experiment_type"]).collect()   
    sns.catplot(data=df,x="experiment_type", kind="count" )  
    del df

def _remove_incomplete_sequences(no_nulls:bool):
    df = load_training_dataset(no_nulls)    
    # create a new df where each sequence appears twice. The assumption is that each occurence
    # `experiment_type` maps to either DMS_MaP or 2A3_MaP    
    df_with_both_experiments = df.filter(pl.int_range(0, pl.count()).over("sequence") < 2)    
    return df_with_both_experiments


@lru_cache
def get_reactivity_error_cols():
    cols = []
    for x in range(1,206+1):
        if x/10 <= 0.9:
            cols.append(f"reactivity_error_000{x}")
        elif x/10 <= 9.9:
            cols.append(f"reactivity_error_00{x}")
        else:
            cols.append(f"reactivity_error_0{x}")
    return cols


@lru_cache
def get_reactivity_cols():
    cols = []
    for x in range(1,206+1):
        if x/10 <= 0.9:
            cols.append(f"reactivity_000{x}")
        elif x/10 <= 9.9:
            cols.append(f"reactivity_00{x}")
        else:
            cols.append(f"reactivity_0{x}")
    return cols


def float_feature(value):
    """Returns a float_list from a float / double."""
    if value is None:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))    
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))
    

def sequence_to_ndarray(value):
    seq = [s.encode() for s in list(value)]
    lookup = string_lookup()
    encoded_sequence = lookup(seq)
    encoded_sequence = encoded_sequence.numpy()    
    encoded_sequence = np.concatenate((encoded_sequence, np.zeros(MAX_LEN-encoded_sequence.shape[0]))).astype(np.float32)
    return encoded_sequence
    

def sequence_feature(value):
    """Returns a bytes_list from a list of byte."""
    encoded_sequence = sequence_to_ndarray(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=encoded_sequence.tolist()))    


def string_feature(value):
    """Returns a bytes_list from a string"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def create_example(row:Dict):   
    reactivities = np.asarray([row[rw] for rw in get_reactivity_cols()])   
    feature = {
        "sequence": sequence_feature(row['sequence']),
        "experiment_type": string_feature(row['experiment_type']), 
        "reactivity" : float_feature( np.concatenate(( reactivities, np.zeros(MAX_LEN-reactivities.shape[0]))).astype(np.float32))       
    }      

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "sequence": tf.io.VarLenFeature(dtype=tf.float32),
        "experiment_type": tf.io.VarLenFeature(dtype=tf.string),  
        "reactivity" :  tf.io.VarLenFeature(dtype=tf.float32),         
    }     

    features = tf.io.parse_single_example(example, feature_description)

    # convert from sparse tensor to dense tensor
    out = {}    
    out['sequence'] = tf.sparse.to_dense(features["sequence"])
    out['experiment_type'] = tf.sparse.to_dense(features["experiment_type"])
    out['reactivity'] = tf.sparse.to_dense(features["reactivity"]) 

    return out



def encode_experiment_type(value:str):    
    if value == "DMS_MaP":        
        # between 0.1 and 0.5
        encode = tf.linspace(0.1, 0.5 , 457, axis=-1)
    else:
        # between 0.6 and 1.0
        encode = tf.linspace(0.6, 1.0 , 457, axis=-1)

    return encode


def encode_experiments(x):
    exp_type = x['experiment_type']
    x['experiment_type'] = encode_experiment_type(exp_type)
    return x


def concat_targets(x):        
    encoded_sequence = tf.convert_to_tensor(x['sequence'] , dtype=tf.float32) 
    encoded_sequence = tf.expand_dims(encoded_sequence, axis=0) 
    
    exp = x['experiment_type']
    exp = tf.expand_dims(exp, axis=0)

    targets = tf.convert_to_tensor(x['reactivity'] , dtype=tf.float32)
    targets = tf.clip_by_norm(targets, 0.5)
    targets = tf.expand_dims(targets, axis=0)    
    return (encoded_sequence,exp), targets  


def cast_types(features,targets):    
    (sequences, exp) = features
    sequences = tf.cast(sequences, tf.float32)
    exp = tf.cast(exp,tf.float32)
    targets = tf.cast(targets,tf.float32)

    seq = tf.reshape(sequences,(1,457))
    exp = tf.reshape(exp,(1,457))
    targets = tf.reshape(targets,(1,457))
    return ((seq,exp),targets)


@tf.function
def read_tfrecord_fn(no_nulls=True, training_set=True,threshold=70): 
    """
    Read dateset from filesystem. The default set is the training set which is a percentage of the
    entire dataset provider by the argument `threshold`. To return the validation set, the argument 
    `training_set` should be False

    """   
    path = get_train_tfrecords_path(no_nulls=no_nulls)    
    files = [join(path,f) for f in os.listdir(path)]
    size = len(files)
    
    record_set: None | List = None
    if training_set:
        split_index = int((threshold * size ) // 100) 
        record_set = files[:split_index]
    else:
        split_index = int(( (100 - threshold) * size ) // 100) 
        record_set = files[size-split_index:] 

    del files
    
    raw_dataset:tf.data.TFRecordDataset = tf.data.TFRecordDataset(filenames=record_set,num_parallel_reads=tf.data.AUTOTUNE)   
    ds = raw_dataset.map(parse_tfrecord_fn,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)    
    ds = ds.map(encode_experiments,num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
    ds = ds.map(concat_targets,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)    
    ds = ds.map(cast_types,num_parallel_calls=tf.data.AUTOTUNE)    
    ds = ds.shuffle(1000, reshuffle_each_iteration = True)       
    return ds.repeat().prefetch(tf.data.AUTOTUNE)



def get_log_path(model_name:str) -> str:
    out_model_path = "out/logs"        
    log_path = os.path.join(out_model_path,model_name)
    return log_path


def submission_path():
    path = "out/submission.csv" 
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["id", "reactivity_DMS_MaP", "reactivity_2A3_MaP"]
            writer.writerow(field)
        return path
    else:
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["id", "reactivity_DMS_MaP", "reactivity_2A3_MaP"]
            writer.writerow(field)
        return path       

    

def get_or_create_model_path(model_name:str) -> str:
    current_path = get_model_path(model_name=model_name)
    if os.path.exists(current_path) and os.path.isfile(current_path):
        os.remove(current_path)
        return get_model_path(model_name=model_name)
    else:
        return current_path 


def get_or_create_log_path(model_name:str) -> str:
    current_path = get_log_path(model_name=model_name)
    if os.path.exists(current_path):
        shutil.rmtree(current_path)
        os.mkdir(current_path)
        return get_log_path(model_name=model_name)
    else:
        os.mkdir(current_path)
        return get_log_path(model_name=model_name)   


def get_model_path(model_name:str) -> str:
    out_model_path = "out/models"
    model_name_with_extenstion = f"{model_name}.keras"
    model_path = os.path.join(out_model_path,model_name_with_extenstion)
    return model_path


def load_fs_model(model_name:str):
    out_model_path = get_model_path(model_name=model_name)    
    model = tf.keras.saving.load_model(out_model_path)
    return model


def connect_to_database():
    return create_engine("postgresql+psycopg2://rna_submission_user:rna_submission_pwd@localhost/rna_submission_db")