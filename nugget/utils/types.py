from typing import *
from dataclasses import dataclass

from transformers import DynamicCache
import torch
from torch import Tensor


def truncate_pkv(past_kv: DynamicCache, k: int) -> DynamicCache:
    ret = DynamicCache()
    ret.key_cache = [layer[:, :, -k:] for layer in past_kv.key_cache]
    ret.value_cache = [layer[:, :, -k:] for layer in past_kv.value_cache]
    ret.seen_tokens = k
    return ret


def gather_cache(cache: DynamicCache, index: Tensor):
    bsz, n_head, n_token, head_dim = cache.key_cache[0].shape
    n_layer = len(cache.key_cache)
    ret = DynamicCache()
    # index shape (bsz, nugget) -> (bsz, head, nugget, head_dim)
    index_exp = index[:, None, :, None].expand(bsz, n_head, -1, head_dim)
    for i_layer in range(n_layer):
        # kv shape (in LLaMA) (bsz, heads, token, head_dim)
        ret.key_cache.append(cache.key_cache[i_layer].gather(2, index_exp))
        ret.value_cache.append(cache.value_cache[i_layer].gather(2, index_exp))
    ret.seen_tokens = n_token
    return ret


def cat_cache(past_caches: List[DynamicCache]) -> DynamicCache:
    # concatenate key values into one
    past_caches = [cache for cache in past_caches if cache is not None and cache.seen_tokens > 0]
    if len(past_caches) == 0:
        return DynamicCache()
    ret = DynamicCache()
    for i_layer in range(len(past_caches[0])):
        ret.key_cache.append(torch.cat([cache.key_cache[i_layer] for cache in past_caches], dim=2))
        ret.value_cache.append(torch.cat([cache.value_cache[i_layer] for cache in past_caches], dim=2))
    ret.seen_tokens = ret.key_cache[0].shape[2]
    return ret


@dataclass
class Nuggets:
    """
    `encoding`, shaped as (bsz, #nugget, dim), is the nugget encodings, which is masked by  `mask`, shaped
    as (bsz, #nugget). Note that `mask` is of `bool` type instead of int64.
    `index`, shaped as (bsz, #nugget), is the index of each nugget. It is also masked by `mask`.
    `index_in_batch`, shaped as (bsz, #nugget), is the index of nugget in **this batch**.
    Please note that index is supposed to be the indices of nuggets in the whole sequence, and will
    be derived from `position_ids` if provided. `index_in_batch` is the nugget indices in this batch,
    ignoring context.
    Below are some variables that might not be needed as output but for internal use:
    `scores`, shaped as (bsz, #nugget), is the logits of nuggets. Masked by `mask`.
    `all_scores`, shaped as (bsz, #token), is the logits of all tokens. It should be masked with a mask that
    is not present in this tuple.
    """
    encoding: Optional[Union[Tensor, DynamicCache]]
    mask: Optional[Tensor]
    scores: Optional[Tensor] = None
    index: Optional[Tensor] = None
    all_scores: Optional[Tensor] = None
    index_in_batch: Optional[Tensor] = None

    @property
    def indices(self) -> List[List[int]]:
        selected_indices = list()
        for ma, ti in zip(self.mask, self.index):
            selected_indices.append(ti[ma].cpu().tolist())
        return selected_indices

    @property
    def is2d(self) -> bool:
        return isinstance(self.encoding, DynamicCache)

    @staticmethod
    def cat(nuggets: List["Nuggets"]) -> "Nuggets":
        # concatenate multiple nuggets into one
        nuggets = [nug for nug in nuggets if nug is not None and nug.encoding is not None]
        if len(nuggets) == 0:
            return Nuggets(None, None)
        if len(nuggets) == 1:
            return nuggets[0]
        if not nuggets[0].is2d:
            enc = torch.cat([nug.encoding for nug in nuggets], dim=0)
        else:
            enc = cat_cache([nug.encoding for nug in nuggets])

        def safe_cat(name: str, dim: int = 1) -> Optional[Tensor]:
            items = [getattr(nug, name) for nug in nuggets]
            if any(item is None for item in items):
                return None
            return torch.cat(items, dim=dim)

        return Nuggets(
            encoding=enc, mask=safe_cat('mask'), scores=safe_cat('safe_scores'),
            index=safe_cat('index'), all_scores=safe_cat('all_scores'),
        )

    @property
    def safe_scores(self) -> Tensor:
        # return scores if it exists, otherwise return zero tensors
        if self.scores is not None:
            return self.scores
        if self.is2d:
            float_dtype = self.encoding.key_cache[0].dtype
        else:
            float_dtype = self.encoding.dtype
        return self.mask.new_zeros(self.mask.shape, dtype=float_dtype)

    def valid(self) -> bool:
        # check if shapes are compatible
        bsz, n = self.mask.shape
        if self.is2d:
            encoding_shape = self.encoding.key_cache[0].shape
            if encoding_shape[0] != bsz or encoding_shape[2] != n:
                return False
        else:
            if self.encoding.shape[0] != bsz or self.encoding.shape[1] != n:
                return False
        if self.scores is not None:
            if self.scores.shape[0] != bsz or self.scores.shape[1] != n:
                return False
        if self.index is not None:
            if self.index.shape[0] != bsz or self.index.shape[1] != n:
                return False
        return True

    def __len__(self) -> int:
        return self.mask.shape[1]


@dataclass
class NuggetInspect:
    # For inspection; contains both tokens and nuggets, but only a single instance
    tokens: Union[List[int], List[str]]
    index: List[int]
    scores: List[float]

    def to_tokens(self, tokenizer):
        tokens = tokenizer.convert_ids_to_tokens(self.tokens)
        is_bpe = any(tok.startswith('▁') or tok.startswith('Ġ') for tok in tokens)
        converted = []
        for tok in tokens:
            if is_bpe:
                if tok.startswith('▁') or tok.startswith('Ġ'):
                    converted.append(' ' + tok[1:])
                else:
                    converted.append(tok)
            else:
                if tok.startswith('##'):
                    converted.append(tok[2:])
                else:
                    converted.append(' ' + tok)
        self.tokens = converted
