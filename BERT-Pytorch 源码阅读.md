# BERT-Pytorch 源码阅读

---

## 0. 数据准备

由于只是用来测试，因此，数据随便选了一个文本相似度数据集

## 1.  整体描述

BERT-Pytorch 在分发包时，主要设置了两大功能：

- bert-vocab ：统计词频，token2idx, idx2token 等信息。对应 `bert_pytorch.dataset.vocab` 中的 `build` 函数。
- bert：对应 `bert_pytorch.__main__` 下的 train 函数。

### 1. bert-vocab

```
python3 -m ipdb test_bert_vocab.py  # 调试 bert-vocab
```

其实 bert-vocab 内部并没有什么重要信息，无非就是一些自然语言处理中常见的预处理手段， 自己花个十分钟调试一下就明白了， 我加了少部分注释， 很容易就能明白。

内部继承关系为： 

```
TorchVocab --> Vocab --> WordVocab
```

### 2. bert

#### 1. Bert Model

![整体结构图](.\img\all.png)



