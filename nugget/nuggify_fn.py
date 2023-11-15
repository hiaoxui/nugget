from .adaptors.bert import adapt_bert
from .adaptors.bart import adapt_bart
from .scorer import NuggetScorer


def nuggify(
        model, scorer_layer: int = 3, residual_start: int = 0, residual_end: int = -1,
        value_ffn: bool = True,
):
    """
    :param model: A base Huggingface/Transformer model
    :param scorer_layer: The scorer takes the features of the `scorer_layer`-th layer in transformers.
    :param residual_start: See below.
    :param residual_end: The residual connections will be built from encoder to
    layers[residual_start:residual_end] in decoder.
    :param value_ffn: Append a value FFN to the nugget encodings.
    """
    residual_end = residual_end if residual_end > 0 else model.config.num_hidden_layers

    if model.config.model_type in ['bert']:
        adapt_fn = adapt_bert
    elif model.config.model_type in ['bart', 'mbart']:
        adapt_fn = adapt_bart
    else:
        raise NotImplementedError
    feeder, scorer_feat, encoder, decoder = adapt_fn(model, scorer_layer, residual_start, residual_end)
    scorer = NuggetScorer(scorer_feat, model.config.d_model, feeder, value_ffn)
    return scorer, encoder, decoder
