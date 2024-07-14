from typing import Union, Tuple, Optional

from transformers import DynamicCache
import torch

from .adaptors.score_feeder import NuggetScoreFeeder
from .utils.types import Nuggets, gather_cache_tuple, gather_cache_dynamic_cache, truncate_pkv


class NuggetScorer(torch.nn.Module):
    def __init__(
            self, base_transformer, d_model: int, feeder: NuggetScoreFeeder,
            value_ffn: bool, ratio: float, no_grad: bool = False
    ):
        super().__init__()
        self.base_transformer, self.feeder = base_transformer, feeder
        self.ratio, self.no_grad = ratio, no_grad
        # self.base_transformer.requires_grad_(False)
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
            hidden_states: Union[torch.Tensor, DynamicCache, None] = None,
            position_ids: Optional[torch.Tensor] = None, use_cache: bool = False, **kwargs
    ) -> Union[Nuggets, Tuple[Nuggets, Nuggets]]:
        # `position_ids`: The position IDs of the associated tokens. Its length can be longer than
        # `input_ids` as base_transformer can take past key values.
        bsz, seq_len = input_ids.shape

        transformer_kwargs = {
            'output_hidden_states': True, 'input_ids': input_ids, 'attention_mask': attention_mask,
            'use_cache': use_cache,
        }
        transformer_kwargs.update(kwargs)
        if position_ids is not None:
            transformer_kwargs['position_ids'] = position_ids
        transformer_out = self.base_transformer(**transformer_kwargs)

        # attention_mask could be longer than sequence length if past_kv are passed.
        attention_mask = attention_mask[:, -seq_len:]
        if self.no_grad:
            with torch.no_grad():
                scores = self.non_linear(transformer_out.hidden_states[-1]).squeeze(2)
        else:
            scores = self.non_linear(transformer_out.hidden_states[-1]).squeeze(2)
        scores[~attention_mask.to(dtype=torch.bool)] = torch.finfo(scores.dtype).min
        scores_for_selection = scores.detach().clone()
        if self.force_last:
            last_index = torch.clamp(attention_mask.sum(1, keepdim=True)-1, 0, None)
            # the last index of `scores_for_selection` is set as maximum to make sure it will be selected
            scores_for_selection.scatter_(
                1, last_index, scores.new_full([scores.shape[0], 1], torch.finfo(scores.dtype).max)
            )
            # the scores of the last index will be set as 0.  This removes it from the computational graph.
            scores = scores.scatter(1, last_index, scores.new_zeros([scores.shape[0], 1]))

        n_token = attention_mask.sum(dim=1)
        n_nugget = torch.ceil(n_token * self.ratio).to(torch.int64)
        n_nugget[n_nugget == 0] = 1
        n_nugget[n_nugget > n_token] = n_token[n_nugget > n_token]
        max_nugget = n_nugget.max()
        nugget_mask = torch.arange(max_nugget, device=max_nugget.device)[None, :] < n_nugget[:, None]

        sorted_indices = torch.argsort(scores_for_selection, dim=1, descending=True)[:, :max_nugget]
        index_to_resort = sorted_indices.clone()
        index_to_resort[~nugget_mask] = torch.iinfo(index_to_resort.dtype).max
        indices = sorted_indices.gather(1, index_to_resort.argsort(1))

        if isinstance(hidden_states, torch.Tensor):
            enc = hidden_states.gather(1, indices[:, :, None].expand(-1, -1, hidden_states.shape[2]))
            if self.value_ffn is not None:
                enc = self.value_ffn(enc)
        elif isinstance(hidden_states, tuple):
            # is decoder-only models
            enc = gather_cache_tuple(hidden_states, indices)
        elif isinstance(hidden_states, DynamicCache):
            enc = gather_cache_dynamic_cache(hidden_states, indices)
        nugget_scores = scores.gather(1, indices)
        if position_ids is None:
            nugget_index = indices
        else:
            nugget_index = position_ids[:, -seq_len:].gather(1, indices)

        nuggets = Nuggets(enc, nugget_mask, nugget_scores, nugget_index, scores, index_in_batch=indices)

        if not use_cache:
            return nuggets
        else:
            pkv = truncate_pkv(transformer_out.past_key_values, seq_len)
            return nuggets, Nuggets(pkv, attention_mask)

    def score_context(self, nuggets: Nuggets):
        return self.feeder(nuggets.scores)
    
    def load_scorer(self, path_or_state: str | dict):
        if isinstance(path_or_state, dict):
            states = path_or_state
        else:
            states = torch.load(path_or_state, map_location=torch.device('cpu'))
        if 'non_linear' in states:
            self.non_linear.load_state_dict(states['non_linear'])
            if 'value_ffn' in states:
                self.value_ffn.load_state_dict(states['value_ffn'])
        else:
            self.non_linear.load_state_dict(states)
