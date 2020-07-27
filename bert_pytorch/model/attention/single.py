import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        Args: query, key, value 同源且 shape 相同
            query: [batch_size, head_num, seq_len, dim]
            key: [batch_size, head_num, seq_len, dim]
            value: [batch_size, head_num, seq_len, dim]
        """

        # [batch_size, head_num, seq_len, dim] * [batch_size, head_num, dim, seq_len]
        # = [batch_size, head_num, seq_len, seq_len]
        # 得到的是：序列中每个词在其他词上的注意力权重分布
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            # False=0 ， True=1
            # mask 本来里边pad_index都为False， False==0，则pad_index转为True
            # mask == 0为True，填充scores对应元素为-1e9
            # 只是去除 pad_index 影响，将其softmax分布概率降为0
            scores = scores.masked_fill(mask == 0, -1e9)

        # [batch_size, head_num, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # [batch_size, head_num, seq_len, seq_len] * [batch_size, head_num, seq_len, dim]
        # =[batch_size, head_num, seq_len, dim]
        return torch.matmul(p_attn, value), p_attn
