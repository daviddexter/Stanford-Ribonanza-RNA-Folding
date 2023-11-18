import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class SequenceAndExperimentInputs(tf.keras.layers.Layer):   
    
    def __init__(self, emb_out_dim=64):        
        super(SequenceAndExperimentInputs, self).__init__()

        self.sequence_input = tf.keras.layers.InputLayer(input_shape=(1,457))
        self.experiment_input = tf.keras.layers.InputLayer(input_shape=(1,457))
        self.embedding = tf.keras.layers.Embedding(input_dim=457, 
                                                   output_dim=emb_out_dim,
                                                   mask_zero=True,
                                                   embeddings_initializer=tf.keras.initializers.GlorotUniform())


    def call(self,inputs):        
        seqs = inputs[0]
        exps = inputs[1]

        seqs = self.sequence_input(seqs)
        exps = self.experiment_input(exps)

        seq_embs = self.embedding(seqs)   
        exp_embs = self.embedding(exps)        

        return (seq_embs,exp_embs)
    











        

        

    



