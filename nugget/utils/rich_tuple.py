from typing import *

import torch
from ..common import Nuggets


class PastKeyValues(tuple):
    # A tuple of tuple (size 2) of tensors, shape (layer, 2) x (bsz, head, nugget, head_dim)
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


def combine_past_kv(past_kvs: List[Union[Nuggets, PastKeyValues]]) -> PastKeyValues:
    past_kvs = [pkv.encoding if isinstance(pkv, Nuggets) else pkv for pkv in past_kvs]
    if len(past_kvs) == 1:
        return past_kvs[0]
    ret = []
    for i_layer in range(len(past_kvs)):
        layer = []
        for kv in range(2):
            layer.append(torch.cat([past_kv[i_layer][kv] for past_kv in past_kvs], dim=2))
        ret.append(tuple(layer))
    return PastKeyValues(ret)
