import csv
from sqlalchemy import Column, text

import tensorflow as tf
from loguru import logger
from tqdm import tqdm
from sqlalchemy.orm import Session

from keras.src.engine import data_adapter

from rna_model.model.layers import MultiHeadAttentionBlock, SelfAttentionBlock, SequenceAndExperimentInputs
from rna_model.utils import connect_to_database, encode_experiment_type, get_or_create_log_path, get_or_create_model_path, load_testdataset, read_tfrecord_fn, sequence_to_ndarray, submission_path

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import INTEGER,NUMERIC

class Base(DeclarativeBase):
    pass


class Submissions(Base):
    __tablename__ = "submissions"

    id = Column(INTEGER, primary_key=True)
    reactivity_dms_map = Column(NUMERIC)
    reactivity_2a3_map = Column(NUMERIC)



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


    # def get_config(self):
    #     config = super().get_config()
    #     config.update(
    #         {
    #             "inputs": self.inputs,
    #             "mhatt1": self.mhatt1,
    #             "mhatt2": self.mhatt2,
    #             "mhatt3": self.mhatt3,
    #             "mhatt4": self.mhatt4,
    #             "avg1": self.avg1,
    #             "sattn1": self.sattn1,
    #             "sattn2": self.sattn2,
    #             "sattn3": self.sattn3,
    #             "sattn4": self.sattn4,
    #             "dense1": self.dense1,
    #             "dense2": self.dense2,
    #         }
    #     )
    #     return config  
        

    def call(self, inputs):     
        assert isinstance(inputs,tuple) is True
        x = self.inputs(inputs) 
        
        assert x[0] is not None
        assert x[1] is not None        

        x1 = self.mhatt1((x[0],x[1]))
        assert x1.shape == (1, 457, 64)

        x2 = self.mhatt1((x[0],x[1]))
        assert x2.shape == (1, 457, 64)

        x2 = self.mhatt1((x[0],x[1]))
        assert x2.shape == (1, 457, 64)

        x3 = self.mhatt1((x[0],x[1]))
        assert x3.shape == (1, 457, 64)

        x4 = self.mhatt1((x[0],x[1]))
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

        logger.info("Predicting ...")

        df = load_testdataset()    

        engine = connect_to_database()       
        with Session(engine) as session:
            rows = [ row for row in df.iter_rows(named=True)]
            for row in tqdm(rows):
                start_index = row['id_min']
                end_index = row['id_max']
                sequence_length = end_index - start_index

                sequence = row['sequence']
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
                results = list(results)   
                results = [ list(r)  for r in results ] 
                results = [ {"id": r[0], "reactivity_dms_map": float(r[1]), "reactivity_2a3_map": float(r[2]) }  for r in results ] 
                session.execute(insert(Submissions),results) 
                session.commit() 