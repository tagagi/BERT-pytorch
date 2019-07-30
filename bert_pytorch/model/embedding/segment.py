import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        """ 3 为 padding_idx, sent_A, sent_B """
        super().__init__(3, embed_size, padding_idx=0)
