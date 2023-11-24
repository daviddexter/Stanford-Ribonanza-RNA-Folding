from rna_model.utils import read_tfrecord_fn
from rna_model.model.layers import MultiHeadAttentionBlock, SelfAttentionBlock, SequenceAndExperimentInputs
from rna_model.model.performer.fast_attention.tensorflow.fast_attention import Attention



def test_dataset_split():
    split_threshold=70
    train_ds = read_tfrecord_fn(threshold=split_threshold)
    val_ds = read_tfrecord_fn(training_set=False,threshold=split_threshold)    

    samples_num1 = train_ds.reduce(0, lambda x,_: x+1).numpy()
    samples_num2 = val_ds.reduce(0, lambda x,_: x+1).numpy()

    assert samples_num1 != samples_num2

    full_count1 = (100 * samples_num1) // split_threshold
    full_count2 = (100 * samples_num2) // (100 - split_threshold)

    assert full_count1 != full_count2


def test_sequence_exp_input_layer():
    ds = read_tfrecord_fn()
    ds = ds.take(1)
    ds = list(ds.as_numpy_iterator())    
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

    x1 = Attention(hidden_size=64,num_heads=8,attention_dropout=0.5)(x[0],x[1])        
    assert x1.shape == (1, 457, 64)

    x2 = Attention(hidden_size=64,num_heads=8,attention_dropout=0.5)(x[0],x[0])    
    assert x2.shape == (1, 457, 64)


def test_multiheadattentionblock():
    ds = read_tfrecord_fn()
    ds = ds.take(1)
    ds = list(ds.as_numpy_iterator())
    inputs = ds[0][0]

    layer = SequenceAndExperimentInputs()
    x = layer(inputs)

    x = MultiHeadAttentionBlock(name="Multi-Head-1",hidden_size=64,att_heads=2,
                                att_dropout=0.5)(x)
    assert x.shape == (1, 457, 64)


def test_selfattentionblock():
    ds = read_tfrecord_fn()
    ds = ds.take(1)
    ds = list(ds.as_numpy_iterator())
    inputs = ds[0][0]

    layer = SequenceAndExperimentInputs()
    x = layer(inputs)

    x = MultiHeadAttentionBlock(name="Multi-Head-1",hidden_size=64,att_heads=2,
                                att_dropout=0.5)(x) 

    x = SelfAttentionBlock(name="Self-Attention-1",hidden_size=64,att_heads=2,att_dropout=0.5)(x)
    assert x.shape == (1, 457, 64)   