import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class SequenceAndExperimentInputs(tf.keras.layers.Layer):   
    
    def __init__(self):        
        super(SequenceAndExperimentInputs, self).__init__()

        self.sequence_input = tf.keras.layers.InputLayer(input_shape=(1,457))
        self.experiment_input = tf.keras.layers.InputLayer(input_shape=(1,1))
        self.embedding = tf.keras.layers.Embedding(input_dim=(1,457), 
                                                   output_dim=64,
                                                   embeddings_initializer=tf.keras.initializers.GlorotUniform())

    def build(self, input_shape):
        # self.add_weight()

        super(SequenceAndExperimentInputs, self).build(input_shape)

    def call(self,inputs):
        import ipdb;ipdb.set_trace()
        seqs = inputs[0]
        exps = inputs[1]

        seqs = self.sequence_input(seqs)
        exps = self.experiment_input(exps)

        return tf.zeros([3,4],tf.float32)
    











        

        

    



