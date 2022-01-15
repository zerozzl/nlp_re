# 自然语言处理-关系抽取

对比常见模型在关系抽取任务上的效果，主要涉及以下几种模型：

- [Relation Classification via Convolutional Deep Neural Network](https://aclanthology.org/C14-1220.pdf)
- [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034.pdf)
- [Relation Classification via Multi-Level Attention CNNs](https://aclanthology.org/P16-1123.pdf)
- [Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme](https://arxiv.org/pdf/1706.05075.pdf)
- [Joint entity recognition and relation extraction as a multi-head selection problem](https://arxiv.org/pdf/1804.07847.pdf)
- [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://arxiv.org/pdf/1909.03227.pdf)
- [Neural Relation Extraction with Selective Attention over Instances](https://aclanthology.org/P16-1200.pdf)
- [Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions](https://aclanthology.org/N19-1288.pdf)

## Pipline

### Char-level 效果

#### CNN

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|duie|0.916|0.823|0.775|<b>0.92</b>|0.878|0.847|
|ccks2019|0.051|0.041|0.037|<b>0.077</b>|0.066|0.049|

#### LSTM

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|duie|<b>0.947</b>|0.937|0.905|0.943|0.941|0.903|
|ccks2019|0.077|<b>0.086</b>|0.054|0.06|0.076|0.047|

#### Multi-level Attention CNN

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|duie|0.625|0.478|0.455|<b>0.726</b>|0.55|0.523|
|ccks2019|0.028|0.028|0.028|0.028|0.028|0.028|

### Word-level 效果

#### CNN

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|duie|0.88|<b>0.885</b>|0.796|
|ccks2019|<b>0.073</b>|0.063|0.043|

#### LSTM

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|duie|0.913|<b>0.928</b>|0.844|
|ccks2019|<b>0.094</b>|0.077|0.062|

#### Multi-level Attention CNN

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|duie|0.673|<b>0.68</b>|0.609|
|ccks2019|0.028|0.028|0.028|

## Joint

### Char-level 效果

#### Novel Tag

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Rand + CRF|Embed Rand + Bigram + CRF|
|----|----|----|----|----|----|----|
|duie|0.481|0.341|0.317|0.456|0.552|<b>0.571</b>|
|ccks2019|<b>0.162</b>|0.107|0.099|0.098|0.155|0.143|

#### Multi head selection

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Rand + CRF|Embed Rand + Bigram + CRF|
|----|----|----|----|----|----|----|
|duie|<b>0.608</b>|0.342|0.236|0.523|0.276|0.245|
|ccks2019|<b>0.124</b>|0.121|0.11|0.039|0.026|0.004|

#### CasREL

|-|Finetune|Fix|
|----|----|----|
|duie|<b>0.631</b>|0.202|
|ccks2019|<b>0.211</b>|0|

### Word-level 效果

#### Novel Tag

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + CRF|
|----|----|----|----|----|
|duie|0.398|0.36|0.309|<b>0.489</b>|
|ccks2019|0.09|0.08|0.072|<b>0.128</b>|

#### Multi head selection

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + CRF|
|----|----|----|----|----|
|duie|<b>0.557</b>|0.513|0.463|0.378|
|ccks2019|0.105|<b>0.107</b>|0.101|0.019|

## Distant

### Char-level 效果

#### PCNN-ATT

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|ccks2019|<b>0.098</b>|0.05|0.05|0.065|0.077|0.077|
|cndbpedia|0.262|0.215|0.213|<b>0.279</b>|0.211|0.208|

#### Intra-Inter-Bag-ATT

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|ccks2019|0.049|0.049|0.049|0.049|0.049|0.049|
|cndbpedia|0.118|0.107|0.107|<b>0.125</b>|0.108|0.108|

### Word-level 效果

#### PCNN-ATT

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|ccks2019|0.055|<b>0.06</b>|<b>0.06</b>|
|cndbpedia|0.247|<b>0.25</b>|0.241|

#### Intra-Inter-Bag-ATT

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|ccks2019|0.049|0.049|0.049|
|cndbpedia|0.106|<b>0.107</b>|<b>0.107</b>|
