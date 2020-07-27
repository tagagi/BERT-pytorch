import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        # d_model: 词向量维度 [max_len, d_model]
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # [max_len, 1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # [max_len, 1]
        # 欲求 pos / 10000^(2i/d_model)),先求 1 / 10000^(2i/d_model))
        # 对 10000^(-2i/d_model)) 先取log，再取exp
        # div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # shape: (1, ceil(d_model/2))
        div_term = (-torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model).exp()

        # position: [1,2,3,...512]
        # 对一个position, positional_embed: 向量偶数位[0::2] = sin, 向量奇数位[1::2]  = cos
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # d_model为偶数, 则sin 和 cos 列数相同
        # d_model为奇数, 则cos 比 sin 少一列
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])

        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 一种是反向传播需要被optimizer更新的，称之为parameter
        # 一种是反向传播不需要被optimizer更新，称之为buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pe.require_grad = False 位置向量只跟位置有关，不参与训练，是死的
        # 按输入x的序列长度截取指定位置的位置向量
        # 切片：按维度从前往后选择，未到的维度完整保留，不切片
        # x [batch_size, seq_len, d_model]和pe [1, max_len, d_model] 得pe[batch_size, seq_len, d_model]
        return self.pe[:, :x.size(1)]
