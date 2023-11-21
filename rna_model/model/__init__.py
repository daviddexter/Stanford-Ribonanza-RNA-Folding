import tensorflow as tf
from keras.src.engine import data_adapter

from datetime import datetime

from rna_model.model.layers import SequenceAndExperimentInputs
from rna_model.model.performer.fast_attention.tensorflow.fast_attention import Attention, SelfAttention
from rna_model.utils import read_tfrecord_fn


@tf.keras.saving.register_keras_serializable()
class RNAReacitivityModel(tf.keras.Model):
    def __init__(self,hidden_size=64,att_heads=8, att_dropout=0.5):
        super().__init__()
        self.inputs = SequenceAndExperimentInputs(emb_out_dim=hidden_size)        
        self.mhatt1 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name="Multi-Head-1")
        self.sattn1 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name="Self-Attention-1")
        self.sattn2 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name="Self-Attention-2")
        self.sattn3 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name="Self-Attention-3")
        self.sattn4 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name="Self-Attention-4")
        self.sattn5 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name="Self-Attention-5")
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(64,activation=tf.keras.activations.swish)   
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(64,activation=tf.keras.activations.swish)      
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):     
        assert isinstance(inputs,tuple) is True
        x = self.inputs(inputs) 
        
        assert x[0] is not None
        assert x[1] is not None        

        x = self.mhatt1(x[0],x[1])
        assert x.shape == (1, 457, 64)

        # self attention
        x = self.sattn1(x,x)
        x = self.sattn2(x,x)
        x = self.sattn3(x,x)
        x = self.sattn4(x,x)
        x = self.sattn5(x,x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return self.dense3(x)
    
    
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

    def train_model_runner(self):        
        training_set = self.training_dataset
        validation_set = self.validation_dataset        

        self.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
                     loss=tf.keras.losses.mean_absolute_error)
        
        log_dir = "out/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.fit(training_set,validation_data=validation_set,epochs=10,
                 verbose=1,callbacks=[tensorboard_callback])
        
        self.summary()
        self.save('rna_model.keras')