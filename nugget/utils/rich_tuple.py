from typing import *

import torch


class PastKeyValues(tuple):
    def gather(self, index: torch.Tensor):
        ret = list()
        bsz, n_head, n_token, head_dim = self[0][0].shape
        # index shape (bsz, nugget)
        index_exp = index[:, None, :, None].expand(bsz, n_head, -1, head_dim)
        for layer in self:
            gather_layer = list()
            # kv shape (in LLaMA) (bsz, heads, token, head_dim)
            for item in layer:
                gather_layer.append(item.gather(2, index_exp))
            ret.append(tuple(gather_layer))
        return PastKeyValues(ret)
