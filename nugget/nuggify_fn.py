from .adaptors.bert import adapt_bert
from .adaptors.bart import adapt_bart
from .adaptors.t5 import adapt_t5
from .adaptors.llama import adapt_llama
from .scorer import NuggetScorer
from .adaptors.score_feeder import NuggetScoreFeeder


def nuggify(
        model, scorer_layer: int = 3, residual_start: int = 0, residual_end: int = -1,
        value_ffn: bool = True, straight_through: bool = True,
):
    """
    :param model: A base Huggingface/Transformer model
    :param scorer_layer: The scorer takes the features of the `scorer_layer`-th layer in transformers.
    :param residual_start: See below.
    :param residual_end: The residual connections will be built from encoder to
    layers[residual_start:residual_end] in decoder.
    :param value_ffn: Append a value FFN to the nugget encodings.
    :param straight_through: If True, will subtract the score value from the forward pass; otherwise
    the nugget score could affect the forward pass.
    """
    residual_end = residual_end if residual_end > 0 else model.config.num_hidden_layers

    if model.config.model_type in ['bert']:
        adapt_fn = adapt_bert
        hidden_size = model.config.d_model
    elif model.config.model_type in ['bart', 'mbart']:
        adapt_fn = adapt_bart
        hidden_size = model.config.d_model
    elif model.config.model_type in ['t5']:
        adapt_fn = adapt_t5
        hidden_size = model.config.d_model
    elif model.config.model_type in ['llama']:
        adapt_fn = adapt_llama
        hidden_size = model.config.hidden_size
    else:
        raise NotImplementedError
    feeder = NuggetScoreFeeder(straight_through, enable=True)
    feeder, scorer_feat, encoder, decoder = adapt_fn(feeder, model, scorer_layer, residual_start, residual_end)
    scorer = NuggetScorer(scorer_feat, hidden_size, feeder, value_ffn)
    return scorer, encoder, decoder
