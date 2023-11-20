import logging

from transformers import AutoTokenizer, AutoConfig


def load_tokenizer(pretrained: str, **kwargs):
    tokenizer_base_logger = logging.getLogger('transformers.tokenization_utils_base')
    tokenizer_base_logger.setLevel('ERROR')
    config = AutoConfig.from_pretrained(pretrained)
    if config.model_type == 't5':
        kwargs['model_max_length'] = 999999
        kwargs['legacy'] = False
    return AutoTokenizer.from_pretrained(pretrained, **kwargs)
