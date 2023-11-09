from typing import *
from copy import deepcopy

import torch
from transformers import PreTrainedModel

from .adaptors.bert import adapt_bert
from .adaptors.bart import adapt_bart
from .adaptors.score_feeder import NuggetScoreFeeder
from .scorer import NuggetScorer


def nuggify(model, scorer_layer: int = 3, residual_start: int = 0, residual_end: int = -1):
    """
    :param model: A base Huggingface/Transformer model
    """
    residual_end = residual_end if residual_end > 0 else model.config.num_hidden_layers

    if model.config.model_type in ['bert']:
        adapt_fn = adapt_bert
    elif model.config.model_type in ['bart', 'mbart']:
        adapt_fn = adapt_bart
    else:
        raise NotImplementedError
    feeder, scorer_feat, encoder, decoder = adapt_fn(model, scorer_layer, residual_start, residual_end)
    scorer = NuggetScorer(scorer_feat, model.config.d_model, feeder)
    return scorer, encoder, decoder
