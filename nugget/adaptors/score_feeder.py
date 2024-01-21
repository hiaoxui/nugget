from typing import *

import torch


class NuggetScoreFeeder:
    def __init__(self, straight_through: bool, enable: bool = True):
        self.straight_through, self._enable = straight_through, enable
        self.scores: Optional[torch.Tensor] = None

    def __call__(self, scores: Optional[torch.Tensor]):
        if self._enable and scores is not None:
            if self.straight_through:
                self.scores = scores - scores.detach()
            else:
                self.scores = scores
        return self

    def enable(self, option: bool):
        self._enable = option
        if not option:
            self.scores = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scores = None
