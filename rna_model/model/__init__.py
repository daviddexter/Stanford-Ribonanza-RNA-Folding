from functools import lru_cache
from sqlalchemy import Column, asc, between, func, select, text

import tensorflow as tf
from loguru import logger
from tensorflow.python.eager.context import gc
from tqdm import tqdm
from sqlalchemy.orm import Session

from keras.src.engine import data_adapter

from rna_model.model.layers import MultiHeadAttentionBlock, SelfAttentionBlock, SequenceAndExperimentInputs
from rna_model.utils import connect_to_database, encode_experiment_type, get_model_path, get_or_create_log_path, get_or_create_model_path, load_testdataset, read_tfrecord_fn, sequence_to_ndarray, submission_path

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import INTEGER,NUMERIC,TEXT

class Base(DeclarativeBase):
    pass


class Submissions(Base):
    __tablename__ = "submissions"

    id = Column(INTEGER, primary_key=True)
    reactivity_dms_map = Column(NUMERIC)
    reactivity_2a3_map = Column(NUMERIC)


class TestDatasetDump(Base):
    __tablename__ = "test_dataset_dump"

    id_min = Column(INTEGER, primary_key=True)
    id_max = Column(INTEGER, primary_key=True)
    sequence_str = Column(TEXT)
    


@lru_cache
@tf.keras.saving.register_keras_serializable()
class RNAReacitivityModel(tf.keras.Model):
    def __init__(self,hidden_size=64,att_heads=8, att_dropout=0.5):
        super().__init__()
        self.inputs = SequenceAndExperimentInputs(emb_out_dim=hidden_size)        

        self.mhatt1 = MultiHeadAttentionBlock(name="Multi-Head-1",hidden_size=hidden_size,
                                              att_heads=att_heads//2 ,att_dropout=att_dropout)   

        self.mhatt2 = MultiHeadAttentionBlock(name="Multi-Head-2",hidden_size=hidden_size,
                                              att_heads=att_heads,att_dropout=att_dropout) 
        

        self.mhatt3 = MultiHeadAttentionBlock(name="Multi-Head-3",hidden_size=hidden_size,
                                              att_heads=att_heads*2,att_dropout=att_dropout)
        

        self.mhatt4 = MultiHeadAttentionBlock(name="Multi-Head-4",hidden_size=hidden_size,
                                              att_heads=att_heads*4,att_dropout=att_dropout)       

        self.avg1 = tf.keras.layers.Average()
        
        self.sattn1 = SelfAttentionBlock(name="Self-Attention-1",hidden_size=hidden_size,
                                         att_heads=att_heads//2,att_dropout=att_dropout)
        
        self.sattn2 = SelfAttentionBlock(name="Self-Attention-2",hidden_size=hidden_size,
                                         att_heads=att_heads,att_dropout=att_dropout)
        
        self.sattn3 = SelfAttentionBlock(name="Self-Attention-3",hidden_size=hidden_size,
                                         att_heads=att_heads*2,att_dropout=att_dropout)
        
        self.sattn4 = SelfAttentionBlock(name="Self-Attention-4",hidden_size=hidden_size,
                                         att_heads=att_heads*4,att_dropout=att_dropout)    
        
        self.dense1 = tf.keras.layers.Dense(64,activation=tf.keras.activations.relu,
                                            kernel_regularizer=tf.keras.regularizers.L2(0.001))      
        self.dense2 = tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.L2(0.001))    
        

    def call(self, inputs):     
        assert isinstance(inputs,tuple) is True
        x = self.inputs(inputs) 
        
        assert x[0] is not None
        assert x[1] is not None        

        x1 = self.mhatt1((x[0],x[1]))
        assert x1.shape == (1, 457, 64)

        x2 = self.mhatt2((x[0],x[1]))
        assert x2.shape == (1, 457, 64)        

        x3 = self.mhatt3((x[0],x[1]))
        assert x3.shape == (1, 457, 64)

        x4 = self.mhatt4((x[0],x[1]))
        assert x4.shape == (1, 457, 64)        

        x5 = tf.math.multiply(x1,x3)
        x6 = tf.math.multiply(x2,x4)            
           
        avg = self.avg1([x5, x6])       

        # self attention
        x = self.sattn1(avg)
        x = self.sattn2(x)
        x = self.sattn3(x)
        x = self.sattn4(x)
        
        # regressor
        x = self.dense1(x)      
        return self.dense2(x)
    
    
    def train_step(self, data):
        inpts,targets  = data        
        assert isinstance(inpts,tuple) is True

        with tf.GradientTape() as tape:
            assert inpts[0] is not None
            assert inpts[1] is not None            
            targets_pred = self(inpts, training=True)
            loss = self.compute_loss(y=targets, y_pred=tf.reshape(targets_pred, targets.shape))            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, targets_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    

    def test_step(self, data):
        inpts, targets, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        targets_pred = self(inpts, training=False)
        self.compute_loss(inpts, targets, tf.reshape(targets_pred, targets.shape) )
        return self.compute_metrics(inpts, targets, targets_pred,sample_weight)


    @property
    def training_dataset(self):
        return read_tfrecord_fn(True)

    @property
    def validation_dataset(self):
        return read_tfrecord_fn(True,training_set=False)     


    def train_model_runner(self,model_name:str,epochs:int = 30): 
        logger.info("Training model ...")
        out_model_path = get_or_create_model_path(model_name=model_name)
        out_log_path = get_or_create_log_path(model_name=model_name)

        training_set = self.training_dataset
        validation_set = self.validation_dataset    

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0005,decay_steps=1000,decay_rate=0.96,staircase=True)
    

        self.compile(optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule),
                     loss=tf.keras.losses.mean_absolute_error,
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])       

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_log_path, histogram_freq=1)

        self.fit(training_set,validation_data=validation_set,epochs=epochs,steps_per_epoch=1000,
                 validation_steps=1000,verbose=1,callbacks=[tensorboard_callback])
        
        self.summary()
        self.save(out_model_path)
        

        def predict_func(row):
            start_index = row.id_min
            end_index = row.id_max
            sequence_length = end_index - start_index

            sequence = row.sequence_str
            # convert to numpy feature
            encoded_sequence = sequence_to_ndarray(sequence)
            encoded_sequence = tf.expand_dims(encoded_sequence, axis=0)
            encoded_sequence = tf.reshape(encoded_sequence,(1,457))

            # encode DMS_MaP
            exp_dms_map_encoded = encode_experiment_type("DMS_MaP")
            exp_dms_map_encoded = tf.expand_dims(exp_dms_map_encoded, axis=0)
            exp_dms_map_encoded = tf.reshape(exp_dms_map_encoded,(1,457))

            # encode 2A3_MaP
            exp_2a3_map_encoded = encode_experiment_type("2A3_MaP")
            exp_2a3_map_encoded = tf.expand_dims(exp_2a3_map_encoded, axis=0)
            exp_2a3_map_encoded  = tf.reshape(exp_2a3_map_encoded,(1,457))

            exp_dms_predict_dataset = (encoded_sequence,exp_dms_map_encoded)            
            exp_dms_prediction = self.predict(exp_dms_predict_dataset,verbose=0,batch_size=1)
            prediction1 = []
            for i in range(0,sequence_length+1):
                prediction1.append(exp_dms_prediction[0][i][0])


            exp_2a3_predict_dataset = (encoded_sequence,exp_2a3_map_encoded)
            exp_2a3_prediction = self.predict(exp_2a3_predict_dataset,verbose=0,batch_size=1)
            prediction2 = []
            for i in range(0,sequence_length+1):
                prediction2.append(exp_2a3_prediction[0][i][0])

            ids = [ i for i in range(start_index,end_index+1)]

            # merge predictions
            results = zip(ids,prediction1,prediction2)
            del prediction1
            del prediction2
            del ids
            results = list(results)   
            results = [ list(r)  for r in results ] 
            results = [ {"id": r[0], "reactivity_dms_map": float(r[1]), "reactivity_2a3_map": float(r[2]) }  for r in results ]
            return results 
        

        def persist_predictions(results):
            with Session(connect_to_database()) as session:
                session.execute(insert(Submissions),results) 
                del results
                session.commit() 
            
        
        offset_value = 0
        test_size = get_size_of_test_dataset()        
        while offset_value <= test_size:
            dataset = retrieve_date_chunk(offset_value)            
            predictions = [ predict_func(row[0]) for row in tqdm(dataset,desc="Running predictions")]
            del dataset
            # persist to DB            
            predictions = [ persist_predictions(row) for row in tqdm(predictions,desc="Persisting predictions to DB")]                      
            del predictions            
            offset_value+=1000            
            logger.info(f"Done predicting and persisting for {offset_value} records")
            gc.collect()




@tf.keras.saving.register_keras_serializable()
class RNAReacitivityFunctionalModel(tf.keras.Model):
    # def compile_from_config(self, config):       
    #     optimizer = tf.keras.utils.deserialize_keras_object(config["model_optimizer"])
    #     loss_fn = tf.keras.utils.deserialize_keras_object(config["loss_fn"])
    #     metrics = tf.keras.utils.deserialize_keras_object(config["metric"])

    #     # Calls compile with the deserialized parameters
    #     self.compile(optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)   

    def test_step(self, data):
        inpts, targets, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        targets_pred = self(inpts, training=False)
        self.compute_loss(inpts, targets, tf.reshape(targets_pred, targets.shape) )
        return self.compute_metrics(inpts, targets, targets_pred,sample_weight)
    

    def train_step(self, data):
        inpts,targets  = data        
        assert isinstance(inpts,tuple) is True

        with tf.GradientTape() as tape:
            assert inpts[0] is not None
            assert inpts[1] is not None            
            targets_pred = self(inpts, training=True)
            loss = self.compute_loss(y=targets, y_pred=tf.reshape(targets_pred, targets.shape))            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, targets_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}




def get_functional_model(emb_out_dim=64,hidden_size=64,att_heads=8, att_dropout=0.5):
    sequence_input = tf.keras.layers.Input(shape=(457))
    experiment_input = tf.keras.layers.Input(shape=(457))

    embedding_seq = tf.keras.layers.Embedding(input_dim=457, 
                                                   output_dim=emb_out_dim,
                                                   mask_zero=True,
                                                   embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                                                   name="embedding-sequence")(sequence_input)
        
    embedding_exp = tf.keras.layers.Embedding(input_dim=457, 
                                                   output_dim=emb_out_dim,
                                                   mask_zero=True,
                                                   embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                                                   name="embedding-experiment")(experiment_input)
    


    mhatt1 = MultiHeadAttentionBlock(name="Multi-Head-1",hidden_size=hidden_size,
                                              att_heads=att_heads//2 ,att_dropout=att_dropout)((embedding_seq,embedding_exp))
    

    mhatt2 = MultiHeadAttentionBlock(name="Multi-Head-2",hidden_size=hidden_size,
                                              att_heads=att_heads,att_dropout=att_dropout)((embedding_seq,embedding_exp))
    


    mhatt3 = MultiHeadAttentionBlock(name="Multi-Head-3",hidden_size=hidden_size,
                                              att_heads=att_heads*2,att_dropout=att_dropout)((embedding_seq,embedding_exp))
        

    mhatt4 = MultiHeadAttentionBlock(name="Multi-Head-4",hidden_size=hidden_size,
                                              att_heads=att_heads*4,att_dropout=att_dropout)((embedding_seq,embedding_exp))
    

    mul1 = tf.keras.layers.Multiply()([mhatt1,mhatt3])

    mul2 = tf.keras.layers.Multiply()([mhatt2,mhatt4])

    avg1 = tf.keras.layers.Average()([mul1,mul2])

    x = SelfAttentionBlock(name="Self-Attention-1",hidden_size=hidden_size,
                                         att_heads=att_heads//2,att_dropout=att_dropout)(avg1)
    

    x = SelfAttentionBlock(name="Self-Attention-2",hidden_size=hidden_size,
                                         att_heads=att_heads,att_dropout=att_dropout)(x)
        
    x = SelfAttentionBlock(name="Self-Attention-3",hidden_size=hidden_size,
                                         att_heads=att_heads*2,att_dropout=att_dropout)(x)
        
    x = SelfAttentionBlock(name="Self-Attention-4",hidden_size=hidden_size,
                                         att_heads=att_heads*4,att_dropout=att_dropout)(x) 
    

    x = tf.keras.layers.Dense(64,activation=tf.keras.activations.relu,
                                            kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    
    out = tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)

    model = tf.keras.Model(inputs=[sequence_input,experiment_input] , outputs=out)
    # model = RNAReacitivityFunctionalModel(inputs=[sequence_input,experiment_input] , outputs=out)
    return model






def train_runner_functional_model(model_name:str,epochs:int = 30):
    logger.info("Training model ...")
    out_model_path = get_or_create_model_path(model_name=model_name)
    out_log_path = get_or_create_log_path(model_name=model_name)

    training_set = read_tfrecord_fn(True)
    validation_set = read_tfrecord_fn(True,training_set=False)   

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0005,decay_steps=1000,decay_rate=0.96,staircase=True)    

    model = get_functional_model()
    # model.summary()

    model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule),
                     loss=tf.keras.losses.mean_absolute_error,
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])       

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_log_path, histogram_freq=1)

    model.fit(training_set,validation_data=validation_set,epochs=epochs,steps_per_epoch=1000,
                 validation_steps=1000,verbose=1,callbacks=[tensorboard_callback])
    
    logger.info("Saving model ...")  
    model.save_weights(out_model_path,save_format="keras")
    # model.save(out_model_path,save_format="keras")



def predict_func(row,model:tf.keras.Model):
    start_index = row.id_min
    end_index = row.id_max
    sequence_length = end_index - start_index

    sequence = row.sequence_str
    # convert to numpy feature
    encoded_sequence = sequence_to_ndarray(sequence)
    encoded_sequence = tf.expand_dims(encoded_sequence, axis=0)
    encoded_sequence = tf.reshape(encoded_sequence,(1,457))

    # encode DMS_MaP
    exp_dms_map_encoded = encode_experiment_type("DMS_MaP")
    exp_dms_map_encoded = tf.expand_dims(exp_dms_map_encoded, axis=0)
    exp_dms_map_encoded = tf.reshape(exp_dms_map_encoded,(1,457))

    # encode 2A3_MaP
    exp_2a3_map_encoded = encode_experiment_type("2A3_MaP")
    exp_2a3_map_encoded = tf.expand_dims(exp_2a3_map_encoded, axis=0)
    exp_2a3_map_encoded  = tf.reshape(exp_2a3_map_encoded,(1,457))

    exp_dms_predict_dataset = (encoded_sequence,exp_dms_map_encoded)            
    exp_dms_prediction = model.predict(exp_dms_predict_dataset,verbose=0,batch_size=1)
    prediction1 = []
    for i in range(0,sequence_length+1):
        prediction1.append(exp_dms_prediction[0][i][0])


    exp_2a3_predict_dataset = (encoded_sequence,exp_2a3_map_encoded)
    exp_2a3_prediction = model.predict(exp_2a3_predict_dataset,verbose=0,batch_size=1)
    prediction2 = []
    for i in range(0,sequence_length+1):
        prediction2.append(exp_2a3_prediction[0][i][0])

    ids = [ i for i in range(start_index,end_index+1)]

    # merge predictions
    results = zip(ids,prediction1,prediction2)
    del prediction1
    del prediction2
    del ids
    results = list(results)   
    results = [ list(r)  for r in results ] 
    results = [ {"id": r[0], "reactivity_dms_map": float(r[1]), "reactivity_2a3_map": float(r[2]) }  for r in results ]
    return results 


def persist_predictions(results):
    with Session(connect_to_database()) as session:
        session.execute(insert(Submissions),results) 
        del results
        session.commit()



def predict_with_functional_model(model_name:str):
    out_model_path = get_model_path(model_name=model_name)
    # model = tf.keras.saving.load_model(out_model_path)
    model = get_functional_model()
    model.load_weights(out_model_path)     

    offset_value = 0
    test_size = get_size_of_test_dataset()        
    while offset_value <= test_size:
        dataset = retrieve_date_chunk(offset_value)            
        predictions = [ predict_func(row[0],model) for row in tqdm(dataset,desc="Running predictions")]        
        del dataset
        # persist to DB            
        predictions = [ persist_predictions(row) for row in tqdm(predictions,desc="Persisting predictions to DB")]                      
        del predictions            
        offset_value+=1000            
        logger.info(f"Done predicting and persisting for {offset_value} records")
        gc.collect()  

    

def persist_test_dataset():
    # load dataset
    df = load_testdataset()
    engine = connect_to_database()       
    with Session(engine) as session:
        logger.info("Transforming test dataset")
        results = [ {"id_min": row['id_min'], "id_max" : row['id_max'], "sequence_str": row["sequence"]} 
                   for row in df.iter_rows(named=True) ]              
        logger.info("Inserting data into database")
        session.execute(insert(TestDatasetDump),results)
        session.commit()
        logger.info("Done")



def get_size_of_test_dataset():
    with Session(connect_to_database()) as session:        
        stmt = select(func.count(TestDatasetDump.id_min))               
        count = session.execute(stmt)    
    return count.scalar()


def retrieve_date_chunk(offset_value:int):         
    with Session(connect_to_database()) as session:
        stmt = select(TestDatasetDump).order_by(asc(TestDatasetDump.id_min)).fetch(1000).offset(offset=offset_value)
        results = session.execute(stmt).all() 
    return results
            
    