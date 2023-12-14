import torch
from typing import *

from .adaptors.score_feeder import NuggetScoreFeeder
from .common import Nuggets
from .utils.rich_tuple import PastKeyValues


class NuggetScorer(torch.nn.Module):
    def __init__(
            self, base_transformer, d_model: int, feeder: NuggetScoreFeeder,
            value_ffn: bool, ratio: float
    ):
        super().__init__()
        self.base_transformer, self.feeder = base_transformer, feeder
        self.ratio = ratio
        self.base_transformer.requires_grad_(False)
        self.non_linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model, True),
            torch.nn.ReLU(True),
            torch.nn.Linear(d_model, 1, True)
        )
        self.value_ffn = torch.nn.Linear(d_model, d_model, True) if value_ffn else None
        # Force the last token to be nugget; useful for decoder-only transformers
        self.force_last: bool = False

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            hidden_states: Union[torch.Tensor, PastKeyValues], **kwargs
    ) -> Nuggets:
        transformer_out = self.base_transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs
        ).hidden_states[-1]
        scores = self.non_linear(transformer_out).squeeze(-1)
        scores[~attention_mask.to(dtype=torch.bool)] = torch.finfo(scores.dtype).min
        if self.force_last:
            scores = scores.scatter(
                1, attention_mask.sum(1, keepdim=True)-1,
                scores.new_full([scores.shape[0], 1], torch.finfo(scores.dtype).max)
            )

        n_token = attention_mask.sum(dim=1)
        n_nugget = torch.ceil(n_token * self.ratio).to(torch.int64)
        n_nugget[n_nugget == 0] = 1
        n_nugget[n_nugget > n_token] = n_token[n_nugget > n_token]
        max_nugget = n_nugget.max()
        nugget_mask = torch.arange(max_nugget, device=max_nugget.device)[None, :] < n_nugget[:, None]

        sorted_indices = torch.argsort(scores, dim=1, descending=True)[:, :max_nugget]
        index_to_resort = sorted_indices.clone()
        index_to_resort[~nugget_mask] = torch.iinfo(index_to_resort.dtype).max
        indices = sorted_indices.gather(1, index_to_resort.argsort(1))

        if isinstance(hidden_states, torch.Tensor):
            enc = hidden_states.gather(1, indices[:, :, None].expand(-1, -1, hidden_states.shape[2]))
            if self.value_ffn is not None:
                enc = self.value_ffn(enc)
        else:
            # is decoder-only models
            enc = hidden_states.gather(indices)
        nugget_scores = scores.gather(1, indices)

        return Nuggets(enc, nugget_mask, nugget_scores, indices, scores)

    def score_context(self, nuggets: Nuggets):
        return self.feeder(nuggets.scores)
