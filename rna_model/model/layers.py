import tensorflow as tf


from rna_model.model.performer.fast_attention.tensorflow.fast_attention import Attention

@tf.keras.saving.register_keras_serializable()
class SequenceAndExperimentInputs(tf.keras.layers.Layer):   
    
    def __init__(self, emb_out_dim=64):        
        super().__init__()
        self.sequence_input = tf.keras.layers.InputLayer(input_shape=(1,457))
        self.experiment_input = tf.keras.layers.InputLayer(input_shape=(1,457))

        self.embedding_seq = tf.keras.layers.Embedding(input_dim=457, 
                                                   output_dim=emb_out_dim,
                                                   mask_zero=True,
                                                   embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                                                   name="embedding-sequence")
        
        self.embedding_exp = tf.keras.layers.Embedding(input_dim=457, 
                                                   output_dim=emb_out_dim,
                                                   mask_zero=True,
                                                   embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                                                   name="embedding-experiment")

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_input": self.sequence_input,
            "experiment_input": self.experiment_input,
            "embedding_seq": self.embedding_seq,
            "embedding_exp": self.embedding_exp,
        }) 
        return config

    def call(self,inputs):              
        seqs, exps = inputs
        seqs = self.sequence_input(seqs)
        exps = self.experiment_input(exps)
        seq_embs = self.embedding_seq(seqs)   
        exp_embs = self.embedding_exp(exps)        
        return (seq_embs, exp_embs)
    


class MultiHeadAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, name:str, hidden_size=64,att_heads=8, att_dropout=0.5):
        super().__init__()
        self.mhatt1 = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name=name)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)        
        self.dropout = tf.keras.layers.Dropout(0.5) 

    def get_config(self):
        config = super().get_config()
        config.update({
            "mhatt1": self.mhatt1,
            "layernorm": self.layernorm,
            "dropout": self.dropout,
        }) 
        return config     

    def call(self,inputs):
        x1,x2 = inputs[0], inputs[1]
        x = self.mhatt1(x1,x2)
        assert x.shape == (1, 457, 64)
        x= self.layernorm(x)
        return self.dropout(x)
    


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, name:str, hidden_size=64,att_heads=8, att_dropout=0.5):
        super().__init__()
        self.sattn = Attention(hidden_size=hidden_size,num_heads=att_heads,attention_dropout=att_dropout,name=name)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)   
        self.dense = tf.keras.layers.Dense(64,activation=tf.keras.activations.relu,
                                           kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)     
        self.dropout = tf.keras.layers.Dropout(0.5) 


    def get_config(self):
        config = super().get_config()
        config.update({
            "sattn": self.sattn,
            "layernorm1": self.layernorm1,
            "dense":self.dense,
            "layernorm2": self.layernorm2,
            "dropout": self.dropout,
        }) 
        return config

    def call(self,inputs):
        x = self.sattn(inputs,inputs)
        x = self.layernorm1(x)
        x = self.dense(x)
        x = self.dropout(x)
        return self.layernorm2(x)