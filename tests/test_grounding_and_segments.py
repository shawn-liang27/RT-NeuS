"""VLM Yes/No scoring, multi-segment helpers and adaptive sampler fixes.

Run: uv run --no-project --python 3.12 --with pytest --with numpy --with torch --with tqdm --with pillow --with opencv-python-headless python -m pytest tests
"""

import sys
import types

import numpy as np
import pytest
import torch

from neus_core.vlm.yes_no import yes_no_confidence, yes_no_token_ids
from neus_core.segments import find_soft_handover, reconcile_sparse_ltl, resolve_target_window


# ---------------------------------------------------------------- Yes/No scoring

class _Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse = {i: t for t, i in vocab.items()}

    def get_vocab(self):
        return self.vocab

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.inverse[i] for i in ids).replace("▁", " ")


VOCAB = {"▁Yes": 0, "Yes": 1, "▁yes": 2, "YES": 3, "▁No": 4, "no": 5, "▁maybe": 6, "▁Yesterday": 7, "<|im_end|>": 8}


def _logits(probs, size=len(VOCAB)):
    """Logits whose softmax over the vocab equals ``probs`` (missing ids get ~0)."""
    p = np.full(size, 1e-12)
    for idx, value in probs.items():
        p[idx] = value
    return torch.tensor(np.log(p / p.sum()))


def test_only_exact_yes_no_tokens_count():
    # upstream compared stripped token text with "Yes" / "No" exactly (case-sensitive)
    yes_ids, no_ids = yes_no_token_ids(_Tokenizer(VOCAB))
    assert yes_ids == [0, 1]
    assert no_ids == [4]


def test_confidence_is_normalized_over_yes_and_no():
    score = yes_no_confidence(_logits({0: 0.3, 1: 0.1, 4: 0.2, 6: 0.4}), _Tokenizer(VOCAB))
    assert score.confidence == pytest.approx(0.4 / 0.6)
    assert score.mass == pytest.approx(0.6)


def test_detection_follows_the_greedy_token():
    # "maybe" is the top token: not detected even though P(Yes) > P(No), as upstream
    score = yes_no_confidence(_logits({0: 0.3, 4: 0.1, 6: 0.6}), _Tokenizer(VOCAB))
    assert score.response == "maybe" and "yes" not in score.response.lower()
    assert score.confidence == pytest.approx(0.75)
    # a confused, non-compliant answer is visible through the mass
    assert score.mass == pytest.approx(0.4)


def test_no_answer():
    score = yes_no_confidence(_logits({0: 0.1, 4: 0.9}), _Tokenizer(VOCAB))
    assert score.response == "No"
    assert score.confidence == pytest.approx(0.1)


def test_bfloat16_logits():
    score = yes_no_confidence(_logits({0: 0.75, 4: 0.25}).to(torch.bfloat16), _Tokenizer(VOCAB))
    assert score.confidence == pytest.approx(0.75, abs=1e-2)


def _upstream_vllm_score(logits, tokenizer):
    """nsvqa/nsvs/vlm/vllm_client.py @ ff7b16f: greedy 1 token, top-20 logprobs."""
    probs = torch.softmax(logits.float(), dim=-1)
    content = tokenizer.decode([int(probs.argmax())])
    is_detected = "yes" in content.lower()
    top = torch.topk(probs, 20)
    token_prob_map = {tokenizer.decode([int(i)]).strip(): float(p) for p, i in zip(top.values, top.indices)}
    yes_prob, no_prob = token_prob_map.get("Yes", 0.0), token_prob_map.get("No", 0.0)
    if yes_prob + no_prob == 0:
        return is_detected, None  # upstream raised ValueError here
    return is_detected, yes_prob / (yes_prob + no_prob)


@pytest.mark.parametrize("seed", range(200))
def test_matches_upstream_vllm_scoring(seed):
    # one Yes and one No token (upstream overwrote duplicates instead of summing them)
    rng = np.random.default_rng(seed)
    vocab = {"▁Yes": 0, "▁No": 1, "▁yes": 2, "YES": 3, "▁Yesterday": 4, **{f"▁w{i}": 5 + i for i in range(60)}}
    tokenizer = _Tokenizer(vocab)
    logits = torch.tensor(rng.normal(0, 1, len(vocab)))
    logits[rng.integers(0, 5)] += rng.uniform(0, 8)   # often make a yes/no-like token win
    if seed % 2:
        logits[:2] += 3                                # half the cases: Yes/No both inside the top 20
    upstream_detected, upstream_conf = _upstream_vllm_score(logits, tokenizer)
    score = yes_no_confidence(logits, tokenizer)
    assert ("yes" in score.response.lower()) == upstream_detected
    if upstream_conf is None:
        assert score.confidence == 0.0 and score.mass == 0.0  # we score it as "not Yes" instead of crashing
    else:
        assert score.confidence == pytest.approx(upstream_conf, rel=1e-5)


# ---------------------------------------------------------------- multi-segment helpers

def test_handover_requires_q_after_p():
    assert find_soft_handover({100, 103}, {105, 108}, tolerance_frames=5) == {100, 103, 105, 108}
    assert find_soft_handover({100, 103}, {90, 95}, tolerance_frames=10) == set()


def test_handover_same_window_counts():
    # P and Q detected in the same window share frame indices
    assert find_soft_handover({50, 51, 52}, {52}, tolerance_frames=0) == {52}


FPS = 30
INFO = {"fps": FPS, "frame_count": FPS * 600}


def _sec(a, b):
    return set(range(a * FPS, b * FPS, FPS))


def test_reconcile_ignores_reversed_order():
    # Q (90-94s) happens before P (100-104s): no P->Q handover, falls back to the automaton
    groups = reconcile_sparse_ltl(INFO, _sec(100, 105), _sec(90, 95), [], "", BRIDGE_MULT=3, CONTEXT_SECONDS=3, TOLERANCE=8)
    assert groups == [-1]


def test_reconcile_correct_order():
    groups = reconcile_sparse_ltl(INFO, _sec(100, 105), _sec(106, 111), [], "", BRIDGE_MULT=3, CONTEXT_SECONDS=3, TOLERANCE=8)
    assert [(min(g) / FPS, max(g) / FPS) for g in groups] == [(100.0, 110.0)]  # all of Q (106-110s)


@pytest.mark.parametrize("window,expected", [
    ("[start_time - 20, end_time + 1.5]", [-20.0, 1.5]),
    ("[start_time, end_time + 10]", [0.0, 10.0]),
    ("[start_time-5,end_time]", [-5.0, 0.0]),
    ("[start_time - 2.25, end_time + 0.5]", [-2.25, 0.5]),
])
def test_target_window_keeps_decimals(window, expected):
    assert resolve_target_window(window) == expected


# ---------------------------------------------------------------- adaptive sampler

VIDEO_FPS = 30
SCENE = []


class _Batch:
    def __init__(self, arr):
        self.arr = arr

    def asnumpy(self):
        return self.arr


class _VideoReader:
    """Frame i is a 2x2 image filled with SCENE[i] so embeddings follow scene ids."""

    def __init__(self, *_args, **_kwargs):
        pass

    def get_avg_fps(self):
        return VIDEO_FPS

    def __len__(self):
        return len(SCENE)

    def get_batch(self, indices):
        return _Batch(np.stack([np.full((2, 2, 3), SCENE[i], dtype=np.uint8) for i in indices]))


def _cos_sim(a, b):
    a = torch.nn.functional.normalize(torch.as_tensor(a).reshape(1, -1).float(), dim=1)
    b = torch.nn.functional.normalize(torch.as_tensor(b).reshape(-1, a.shape[1]).float(), dim=1)
    return a @ b.T


class _Clip:
    """Text props embed to one direction; frame embeddings depend on the scene id."""

    def __init__(self, text_dir, scene_dirs):
        self.text_dir = torch.tensor(text_dir, dtype=torch.float)
        self.scene_dirs = scene_dirs

    def encode(self, items, convert_to_tensor=True, **_kwargs):
        if isinstance(items[0], str):
            return self.text_dir.repeat(len(items), 1)
        rows = [self.scene_dirs[int(np.asarray(img)[0, 0, 0])] for img in items]
        return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float), dim=1)


@pytest.fixture
def sampler(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=_VideoReader, cpu=lambda _: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(util=types.SimpleNamespace(cos_sim=_cos_sim)))
    sys.modules.pop("rtneus.sampling", None)
    from rtneus import sampling as read_video_adaptive
    yield read_video_adaptive.read_video_adaptive
    sys.modules.pop("rtneus.sampling", None)


def test_relaxed_threshold_fallback_is_used(sampler):
    global SCENE, VIDEO_FPS
    VIDEO_FPS = 30
    SCENE = [0] * (VIDEO_FPS * 20)
    # every frame scores cos=0.20: below threshold 0.21 but above the 0.18 fallback
    clip = _Clip([1.0, 0.0], {0: [0.2, (1 - 0.2 ** 2) ** 0.5]})
    out = sampler(clip, "v.mp4", ["a_cat"], threshold=0.21)
    assert out["video_sample_method"] == "two-stage clip"


def test_dedup_compares_the_right_frames(sampler):
    global SCENE, VIDEO_FPS
    # 10 s video at 2 fps (20 frames): the sampler requests 25 samples, so
    # np.linspace repeats indices; the scene switches every 2 frames
    VIDEO_FPS = 2
    SCENE = [(i // 2) % 2 for i in range(20)]
    clip = _Clip([1.0, 0.0], {0: [1.0, 0.0], 1: [0.0, 1.0]})
    out = sampler(clip, "v.mp4", ["a_cat"], threshold=0.0, tau_r=0.9)
    # each scene block is kept exactly once: frames 0, 2, 4, ..., 18
    assert out["original_indices"] == list(range(0, 20, 2))
