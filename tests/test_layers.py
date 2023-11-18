from rna_model.utils import read_tfrecord_fn
from rna_model.model.layers import SequenceAndExperimentInputs
from rna_model.model.performer.fast_attention.tensorflow.fast_attention import Attention


def test_sequence_exp_input_layer():
    ds = read_tfrecord_fn()
    ds = ds.take(1)
    ds = list(ds.as_numpy_iterator())
    import ipdb;ipdb.set_trace()
    inputs = ds[0][0]
    assert inputs[0].shape == (1,457)
    assert inputs[1].shape == (1,457)
    
    layer = SequenceAndExperimentInputs()
    y = layer(inputs)

    assert y[0].shape == (1, 457, 64)
    assert y[1].shape == (1, 457, 64)

def test_attention_layer():
    ds = read_tfrecord_fn()
    ds = ds.take(1)
    ds = list(ds.as_numpy_iterator())
    inputs = ds[0][0]

    layer = SequenceAndExperimentInputs()
    x = layer(inputs)

    x = Attention(hidden_size=64,num_heads=8,attention_dropout=0.5)(x[0],x[1])
    
    assert x.shape == (1, 457, 64)


    



    