import torch

from .adaptors.score_feeder import NuggetScoreFeeder
from .common import Nuggets


class NuggetScorer(torch.nn.Module):
    def __init__(
            self, base_transformer, d_model: int, feeder: NuggetScoreFeeder,
            value_ffn: bool,
    ):
        super().__init__()
        self.base_transformer, self.feeder = base_transformer, feeder
        self.base_transformer.requires_grad_(False)
        self.non_linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model, True),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, 1, True)
        )
        self.value_ffn = torch.nn.Linear(d_model, d_model, True) if value_ffn else None

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

        if self.value_ffn is not None:
            enc = self.value_ffn(enc)

        return Nuggets(enc, nugget_mask, nugget_scores, indices, scores).sort()

    def score_context(self, nuggets: Nuggets):
        return self.feeder(nuggets.scores)
