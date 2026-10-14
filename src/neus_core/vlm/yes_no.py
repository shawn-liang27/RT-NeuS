"""Yes/No detection scoring shared by every VLM wrapper, following upstream NeuS-QA.

Upstream (vLLM, ``max_tokens=1``, ``temperature=0``, top-20 logprobs) decided
detection from the generated token (``"yes" in token.lower()``) and computed
confidence as P(Yes) / (P(Yes) + P(No)) over the top-20 tokens whose stripped
text is exactly ``Yes`` / ``No`` (a word outside the top 20 counts as 0).
``yes_no_confidence`` reproduces both from the first answer token's logits,
and also reports how much probability the model put on Yes/No at all
(``mass``) so non-compliant answers can be spotted. Where upstream raised an
error (neither word in the top 20) the confidence is 0.
"""

from typing import NamedTuple

import torch

_WORD_MARKERS = ("▁", "Ġ")  # SentencePiece / byte-level BPE leading-space markers
_CACHE_ATTR = "_neus_yes_no_token_ids"


class YesNoScore(NamedTuple):
    response: str       # the greedy (argmax) first token, decoded
    confidence: float   # P(Yes) / (P(Yes) + P(No))
    mass: float         # P(Yes) + P(No); close to 1 when the model follows the Yes/No rule


def _strip(token: str) -> str:
    for marker in _WORD_MARKERS:
        token = token.replace(marker, "")
    return token.strip()


def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Ids of vocabulary tokens whose stripped text is exactly "Yes" / "No" (e.g. ``Yes``, ``▁Yes``)."""
    cached = getattr(tokenizer, _CACHE_ATTR, None)
    if cached is None:
        yes_ids, no_ids = [], []
        for token, token_id in tokenizer.get_vocab().items():
            word = _strip(token)
            if word == "Yes":
                yes_ids.append(token_id)
            elif word == "No":
                no_ids.append(token_id)
        if not yes_ids or not no_ids:
            raise ValueError("Tokenizer has no single-token Yes/No entries")
        cached = (sorted(yes_ids), sorted(no_ids))
        # cached on the tokenizer itself (an id()-keyed dict can hand a new tokenizer a freed one's ids)
        setattr(tokenizer, _CACHE_ATTR, cached)
    return cached


TOP_LOGPROBS = 20  # upstream requested top_logprobs=20 from vLLM


def yes_no_confidence(logits: torch.Tensor, tokenizer, top_k: int = TOP_LOGPROBS) -> YesNoScore:
    """Score a yes/no answer from the logits of its first token.

    Args:
        logits: 1-D vocabulary logits for the first generated position.
        tokenizer: tokenizer whose vocabulary indexes ``logits``.
        top_k: only the ``top_k`` most likely tokens count, as with vLLM logprobs.
    """
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    probs = torch.softmax(logits.float(), dim=-1)
    response = tokenizer.decode([int(probs.argmax())], skip_special_tokens=True).strip()
    top = torch.topk(probs, min(top_k, probs.shape[-1]))
    top_probs = dict(zip(top.indices.tolist(), top.values.tolist()))
    p_yes = sum(top_probs.get(i, 0.0) for i in yes_ids)
    p_no = sum(top_probs.get(i, 0.0) for i in no_ids)
    mass = p_yes + p_no
    confidence = p_yes / mass if mass > 0 else 0.0
    return YesNoScore(response, confidence, mass)
