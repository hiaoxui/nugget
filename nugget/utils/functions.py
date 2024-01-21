from typing import *

from transformers import DynamicCache


def shallow_copy(past_kv: Optional[DynamicCache]) -> DynamicCache:
    ret = DynamicCache()
    if past_kv is not None:
        ret.seen_tokens = past_kv.seen_tokens
        ret.key_cache, ret.value_cache = past_kv.key_cache.copy(), past_kv.value_cache.copy()
    return ret
