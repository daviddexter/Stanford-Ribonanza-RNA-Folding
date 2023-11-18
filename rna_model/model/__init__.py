import tensorflow as tf

from rna_model.model.layers import SequenceAndExperimentInputs
from rna_model.model.performer.fast_attention.tensorflow.fast_attention import Attention,SelfAttention
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
        self.dense = tf.keras.layers.Dense(457)

    def call(self, inputs):        
        inpts = inputs[0][0]
        x = self.inputs(inpts)
        x = self.mhatt1(x)
        x = self.mhatt2(x)
        x = self.mhatt3(x)
        x = self.mhatt4(x)
        out = self.dense(x)



def get_base_model(hidden_size=64,att_heads=8, att_dropout=0.5,emb_out_dim=64):
    sequence_input = tf.keras.layers.Input(shape=(1,457),dtype=tf.float32)
    experiment_input = tf.keras.layers.Input(shape=(1,457),dtype=tf.float32)

    seq_embedding = tf.keras.layers.Embedding(input_dim=457, 
                                          output_dim=emb_out_dim,
                                          mask_zero=True,
                                          embeddings_initializer=tf.keras.initializers.GlorotUniform())(sequence_input) 

    exp_embedding = tf.keras.layers.Embedding(input_dim=457, 
                                          output_dim=emb_out_dim,
                                          mask_zero=True,
                                          embeddings_initializer=tf.keras.initializers.GlorotUniform())(experiment_input)    

   

    x = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)(seq_embedding,exp_embedding)
    x = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)(x,x)
    x = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)(x,x)
    x = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout)(x,x)
    x = tf.keras.layers.Dense(457, activation=tf.keras.activations.relu)(x)
    outputs = tf.keras.layers.Dense(457)(x)
    model = tf.keras.Model((sequence_input,experiment_input),outputs)
    return model
    

    







        
