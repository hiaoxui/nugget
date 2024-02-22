from typing import *

from transformers import DynamicCache
from .types import CacheType


def shallow_copy(past_kv: Union[DynamicCache, CacheType]) -> Optional[CacheType]:
    if isinstance(past_kv, DynamicCache):
        past_kv = past_kv.to_legacy_cache()
    if past_kv is None:
        return

    ret = list()
    for layer in past_kv:
        lay = list()
        for j in range(2):
            lay.append(layer[j].clone())
        ret.append(tuple(lay))
    return tuple(ret)
