import tensorflow as tf

from rna_model.model.layers import SequenceAndExperimentInputs
from rna_model.model.performer.fast_attention.tensorflow.fast_attention import Attention
from rna_model.utils import read_tfrecord_fn


@tf.keras.saving.register_keras_serializable()
class RNAReacitivityModel(tf.keras.Model):
    def __init__(self,hidden_size=64,att_heads=8, att_dropout=0.5):
        super().__init__()
        self.inputs = SequenceAndExperimentInputs(emb_out_dim=hidden_size)
        self.mhatt1 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)
        self.mhatt2 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)
        self.mhatt3 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)
        self.mhatt4 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)
        self.dense1 = tf.keras.layers.Dense(457,activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(457)

    def call(self, inputs):        
        # inpts = inputs[0][0]
        x = self.inputs(inputs)
        x = self.mhatt1(x[0],x[1])
        x = self.mhatt2(x[0],x[0])
        x = self.mhatt3(x[0],x[0])
        x = self.mhatt4(x[0],x[0])
        x = self.dense1(x)
        return self.dense2(x)
    
    def train_step(self, data):
        inpts = data[0][0]
        targets = data[0][1]
        with tf.GradientTape() as tape:
            targets_pred = self(inpts, training=True)
            loss = self.compute_loss(y=targets, y_pred=targets_pred)

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
        inpts = data[0][0]
        targets = data[0][1]

        targets_pred = self(inpts, training=False)
        self.compute_loss(inpts, targets, targets_pred)
        return self.compute_metrics(inpts, targets, targets_pred)        



def train_model_fn():
    training_set = read_tfrecord_fn(True)
    validation_set = read_tfrecord_fn(True,training_set=False)

    model = RNAReacitivityModel()
    model.compile(optimizer=tf.keras.optimizers.AdamW(),loss=tf.keras.losses.mean_absolute_error)
    model.fit(training_set,validation_data=validation_set,epochs=10,verbose=2)



        
