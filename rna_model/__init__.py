import gc
import os
import click
import tensorflow as tf
import polars as pl

from loguru import logger
from rna_model.model import RNAReacitivityModel
from rna_model.utils import ( _remove_incomplete_sequences, create_example, encode_experiment_type, 
                             get_reactivity_cols, 
                             get_reactivity_error_cols, get_train_tfrecords_path, load_fs_model, load_testdataset, 
                             load_training_dataset, read_tfrecord_fn, sequence_to_ndarray)


@click.command
def remove_incomplete_sequences():
    """
    Returns sequences with two experiments carried out; DMS_MaP and 2A3_MaP.
    Instances with more that two experiments are ignored.
    """
    return _remove_incomplete_sequences()   


@click.command
def check_sequence_length_to_reactivitiy_columns():    
    df = _remove_incomplete_sequences()    
    df0 = df.select(pl.col('sequence'))
    df1 = df0.with_columns(sequence_length=pl.col("sequence").map_elements(lambda x: len(x) ))
    df = df1.collect(streaming=True)
    print(df.select(pl.max("sequence_length")))
    print(df.select(pl.min("sequence_length")))


@click.command
def save_reactivity_reactivity_error(): 
    lf = load_training_dataset(no_nulls=True)
    reactivity_cols = get_reactivity_cols()
    reactivity_err_cols = get_reactivity_error_cols()
    lf_reactivity = lf.melt( value_vars=reactivity_cols).select(pl.col('value').alias('reactivity_values'))
    lf_err_reactivity = lf.melt( value_vars=reactivity_err_cols).select(pl.col('value').alias('reactivity_error_values'))
    df_reactivity = lf_reactivity.collect(streaming=True)
    df_err_reactivity = lf_err_reactivity.collect(streaming=True)
    df = df_reactivity.hstack(df_err_reactivity)
    del df_reactivity
    del df_err_reactivity
    print(df.select(pl.min("reactivity_values")))
    print(df.select(pl.max("reactivity_values")))
    print(df.select(pl.median("reactivity_values")))
    print(df.select(pl.std("reactivity_values")))

    print(df.select(pl.min("reactivity_error_values")))
    print(df.select(pl.max("reactivity_error_values")))
    print(df.select(pl.median("reactivity_error_values")))
    print(df.select(pl.std("reactivity_error_values")))

    reactivity_vs_reactivity_err_parquet = "/home/kineticengines/app/datasets/reactivity_vs_reactivity_err.parquet"
    df.write_parquet(reactivity_vs_reactivity_err_parquet,compression_level=20)


@click.command
@click.option("--no-nulls", default=True)
def dump_as_tfrecord(no_nulls:bool):
    lf = _remove_incomplete_sequences(no_nulls=no_nulls) 

    # drop reactivity errors columns    
    cols = get_reactivity_error_cols()
    cols = ",".join(cols)
    lf.drop(cols)

    df = lf.collect(streaming=True)    

    tfrecords_path = get_train_tfrecords_path(no_nulls=no_nulls)

    try:
        os.remove(tfrecords_path)
    except:
        pass 

    count = 1
    for data in df.iter_rows(named=True):
        file_name = f"{count}.tfrecord"
        path = os.path.join(tfrecords_path,file_name)
        
        if count <= 225515:
            logger.info(f"Completed processing {path}")
            count+=1   
            continue

        
        with tf.io.TFRecordWriter(path) as writer:
            example = create_example(data) 
            writer.write(example.SerializeToString())
        del data
        gc.collect()
        logger.info(f"Completed processing {path}")
        count+=1          
    

@click.command
@click.option("--no-nulls", default=True)
def read_tfrecord(no_nulls:bool):
    return read_tfrecord_fn(no_nulls)



@click.command
@click.option('-mn', '--model-name',help="The name of the output model. The out model will be saved in out/models directory.")
@click.option('-ep', '--epochs',help="The number of epochs tp train the model on.")
def train_model(model_name:str,epochs:int):
    if model_name is None:
        click.echo("`model_name` must be specified")
        return  

    if epochs is None:
        click.echo("`epochs` must be specified")
        return

    epochs_count = int(epochs)
    
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = RNAReacitivityModel()
        model.train_model_runner(model_name,epochs_count)