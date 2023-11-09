from typing import *
from copy import deepcopy
from dataclasses import dataclass

import torch

from .adaptors.score_feeder import NuggetScoreFeeder


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
    encoding: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]
    index: Optional[torch.Tensor]
    all_scores: Optional[torch.Tensor] = None

    @property
    def indices(self) -> List[List[int]]:
        selected_indices = list()
        for ma, ti in zip(self.mask, self.index):
            selected_indices.append(ti[ma].cpu().tolist())
        return selected_indices

    def sort(self) -> "Nuggets":
        ars = self.index.argsort(1)
        encoding = self.encoding.gather(1, ars.unsqueeze(2).expand_as(self.encoding))
        mask = self.mask.gather(1, ars)
        scores = self.scores.gather(1, ars)
        index = self.index.gather(1, ars)
        ret = Nuggets(encoding, mask, scores, index, self.all_scores)
        return ret


class NuggetScorer(torch.nn.Module):
    def __init__(self, base_transformer, d_model: int, feeder: NuggetScoreFeeder):
        super().__init__()
        self.base_transformer = base_transformer
        self.feeder = feeder
        self.base_transformer.requires_grad_(False)
        self.non_linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1)
        )

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, hidden_states: torch.Tensor,
            nugget_ratio: float
    ) -> Nuggets:
        transformer_out = self.base_transformer(input_ids, attention_mask)
        scores = self.non_linear(transformer_out.last_hidden_state).squeeze(-1)

        n_token = attention_mask.sum(dim=1)
        n_nugget = torch.ceil(n_token * nugget_ratio).to(torch.int64)
        n_nugget[n_nugget == 0] = 1
        n_nugget[n_nugget > n_token] = n_token[n_nugget > n_token]
        max_nugget = n_nugget.max()
        nugget_mask = torch.arange(max_nugget, device=max_nugget.device)[None, :] < n_nugget[:, None]

        indices = torch.argsort(scores, dim=1, descending=True)[:, :max_nugget]
        enc = hidden_states.gather(1, indices[:, :, None].expand(-1, -1, hidden_states.shape[2]))
        nugget_scores = scores.gather(1, indices)

        return Nuggets(enc, nugget_mask, nugget_scores, indices, scores)

    def score_context(self, nuggets: Nuggets):
        return self.feeder(nuggets.scores)
