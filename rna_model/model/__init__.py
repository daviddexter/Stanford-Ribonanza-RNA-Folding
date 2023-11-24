import os

import tensorflow as tf

from keras.src.engine import data_adapter

from rna_model.model.layers import MultiHeadAttentionBlock, SelfAttentionBlock, SequenceAndExperimentInputs
from rna_model.model.performer.fast_attention.tensorflow.fast_attention import Attention, SelfAttention
from rna_model.utils import read_tfrecord_fn


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
        
        self.dense1 = tf.keras.layers.Dense(64,activation=tf.keras.activations.relu)      
        self.dense2 = tf.keras.layers.Dense(1)
        

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
    

    def _get_model_path(self,model_name:str) -> str:
        out_model_path = "out/models"
        model_name_with_extenstion = f"{model_name}.keras"
        model_path = os.path.join(out_model_path,model_name_with_extenstion)
        return model_path
    
    
    def _get_log_path(self,model_name:str) -> str:
        out_model_path = "out/logs"        
        log_path = os.path.join(out_model_path,model_name)
        return log_path
    

    def _get_or_create_model_path(self,model_name:str) -> str:
        current_path = self._get_model_path(model_name=model_name)
        if os.path.exists(current_path) and os.path.isfile(current_path):
            os.remove(current_path)
            return self._get_model_path(model_name=model_name)
        else:
            return current_path 


    def _get_or_create_log_path(self,model_name:str) -> str:
        current_path = self._get_log_path(model_name=model_name)
        if os.path.exists(current_path):
            os.remove(current_path)
            os.mkdir(current_path)
            return self._get_log_path(model_name=model_name)
        else:
            os.mkdir(current_path)
            return self._get_log_path(model_name=model_name)               


    def train_model_runner(self,model_name:str,epochs:int = 30):   
        out_model_path = self._get_or_create_model_path(model_name=model_name)
        out_log_path = self._get_or_create_log_path(model_name=model_name)

        training_set = self.training_dataset
        validation_set = self.validation_dataset        

        self.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
                     loss=tf.keras.losses.mean_absolute_error)       

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_log_path, histogram_freq=1)

        self.fit(training_set,validation_data=validation_set,epochs=epochs,steps_per_epoch=1000,
                 verbose=1,callbacks=[tensorboard_callback])
        
        self.summary()
        self.save(out_model_path)