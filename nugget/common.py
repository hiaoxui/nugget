from typing import *
from dataclasses import dataclass

import torch

from .utils.rich_tuple import PastKeyValues


@dataclass
class Nuggets:
    """
    `encoding`, shaped as (bsz, #nugget, dim), is the nugget encodings, which is masked by  `mask`, shaped
    as (bsz, #nugget). Note that `mask` is of `bool` type instead of int64.
    `index`, shaped as (bsz, #nugget), is the index of each nugget.
    It is also masked by `mask`.
    Below are some variables that might not be needed as output but for internal use:
    `scores`, shaped as (bsz, #nugget), is the logits of nuggets. Masked by `mask`.
    `all_scores`, shaped as (bsz, #token), is the logits of all tokens. It should be masked with a mask that
    is not present in this tuple.
    """
    encoding: Optional[Union[torch.Tensor, PastKeyValues]]
    mask: Optional[torch.Tensor]
    scores: Optional[torch.Tensor] = None
    index: Optional[torch.Tensor] = None
    all_scores: Optional[torch.Tensor] = None

    @property
    def indices(self) -> List[List[int]]:
        selected_indices = list()
        for ma, ti in zip(self.mask, self.index):
            selected_indices.append(ti[ma].cpu().tolist())
        return selected_indices
