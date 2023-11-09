from typing import *

import torch


class NuggetScoreFeeder:
    def __init__(self):
        self.scores: Optional[torch.Tensor] = None

    def __call__(self, scores: torch.Tensor):
        self.scores = scores
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scores = None
