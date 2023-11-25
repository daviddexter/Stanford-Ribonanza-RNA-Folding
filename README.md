# RNA MODEL

## Versions

### `rna_model`

This is the base experimental model. We stack multihead attention then follow with subsequent self-attention layers.

[Model architecture](https://github.com/daviddexter/Stanford-Ribonanza-RNA-Folding/blob/7337160c415caf83a30f43d333c999218c193b0b/rna_model/model/__init__.py)

Conclusion : NOT IDEAL



### `rna_model2`

In this model, we increase the number of multihead attention layers by introducing `MultiHeadAttentionBlock`. For each `MultiHeadAttentionBlock` we use
varied `attention_heads`. The blocks include a normalization layer. The output of multi-head attentions are multiplied and average then passed as input to
the self-attention block. Self-attention block similarly employs varied `attention_heads`. For the dense layers in this version, we use `relu` instead of
`swish`


[Model architecture](https://github.com/daviddexter/Stanford-Ribonanza-RNA-Folding/blob/6f37ac1cd97d39e1b2d76deeafc0b3a7d264f46b/rna_model/model/__init__.py)

Conclusion : NOT IDEAL


### `rna_model3`

- `rna_model3_234` : Same model as `rna_model2`. We train it over 500 epochs. The hypothesis is that over the large number of epochs, the curve will
smoothen. We also introduce an `ExponentialDecay` learning rate schedule.

-  `rna_model3_500` : Learning rate decay after 1000 iterations; Conclusion : NOT IDEAL

- `rna_model3_500_nadam`  : Use `nadam` as optimizer with the same learning rate schedule. We also employ `L2` kernel 
regularizer of value 0.001

- `rna_model3_500_adamw_l2`: Use `adamW` with the same learning rate. We also employ `L2` kernel regularizer of value 0.001


## References

- https://devennn.github.io/2020/self-and-multihead-attention/
