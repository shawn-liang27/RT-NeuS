"""Engine, presets and configs, with a scripted detector instead of InternVL.

Needs stormpy (prebuilt wheel), so run with:
    uv run --no-project --python 3.12 --with pytest --with numpy --with torch --with tqdm \
        --with pillow --with opencv-python-headless --with stormpy python -m pytest tests
"""

import glob
import json
import os
import random
import subprocess
import sys
import types

import numpy as np
import pytest

pytest.importorskip("stormpy")

from neus_core.config import PipelineConfig  # noqa: E402
from neus_core.pipeline import LocalizationPipeline, run_localization  # noqa: E402
from neus_core.vlm.obj import DetectedObject  # noqa: E402
from neusqa.frame_validator import FrameValidator  # noqa: E402
from rtneus.frame_validator import StageFrameValidator  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The pre-restructure code lives in the NeuS-QA repo (branch neus-improve-efficiency):
#   git clone -b neus-improve-efficiency https://github.com/shawn-liang27/NeuS-QA.git
#   NEUSQA_REPO=/path/to/NeuS-QA python -m pytest
OLD_CODE_COMMIT = "4f07cc8"
OLD_CODE_REPO = os.environ.get("NEUSQA_REPO", REPO)
FPS = 30


class ScriptedDetector:
    """Detects a proposition in a window if any frame of the window lists it in ``schedule``."""

    def __init__(self, schedule, probability=0.9):
        self.schedule = schedule
        self.probability = probability

    def _detected(self, prop, indices):
        return any(prop in self.schedule.get(i, ()) for i in indices)

    def batch_detect(self, seq_of_frames, scene_descriptions, threshold):
        indices = [int(f[0, 0, 0]) * 1000 + int(f[0, 0, 1]) for f in seq_of_frames]
        return [self.detect(seq_of_frames, prop, threshold, indices) for prop in scene_descriptions]

    def detect(self, seq_of_frames, scene_description, threshold, indices=None):
        if indices is None:
            indices = [int(f[0, 0, 0]) * 1000 + int(f[0, 0, 1]) for f in seq_of_frames]
        hit = self._detected(scene_description, indices)
        p = self.probability if hit else 0.0
        return DetectedObject(name=scene_description, is_detected=hit, confidence=p, probability=p)


def _video(seconds, step_seconds=1):
    """One sampled frame per ``step_seconds``; each image encodes its frame index."""
    indices = list(range(0, seconds * FPS, step_seconds * FPS))
    images = [np.full((2, 2, 3), (i // 1000, i % 1000, 0), dtype=np.uint16) for i in indices]
    return {"images": images, "original_indices": indices,
            "video_info": {"fps": FPS, "frame_count": seconds * FPS, "sample_rate": 1}}


def _schedule(spans):
    """{prop: [(start_s, end_s), ...]} -> {frame_idx: {props}}"""
    out = {}
    for prop, ranges in spans.items():
        for start, end in ranges:
            for i in range(start * FPS, end * FPS + 1, FPS):
                out.setdefault(i, set()).add(prop)
    return out


def _sec(segments):
    return segments if segments == [-1] else [(round(s / FPS), round(e / FPS)) for s, e in segments]


# ---------------------------------------------------------------- engine

def test_until_spec_uses_intersect_and_finds_handover():
    detector = ScriptedDetector(_schedule({"a": [(30, 44)], "b": [(45, 50)]}))
    out = run_localization(_video(120), ["a", "b"], '"a" U "b"', "[start_time, end_time]",
                           detector, StageFrameValidator, PipelineConfig(progress_bar=False))
    assert out.segment_mode == "intersect"
    assert len(out.detections) == 2 and out.detections[0] and out.detections[1]
    (start, end), = _sec(out.segments)
    assert 30 <= start <= 45 and 45 <= end <= 53


def test_eventually_spec_uses_union_and_keeps_each_event():
    # sequence question: three tools used at very different times
    detector = ScriptedDetector(_schedule({"glue_gun": [(10, 12)], "needle": [(60, 62)], "scissors": [(110, 112)]}))
    out = run_localization(_video(150), ["glue_gun", "needle", "scissors"],
                           'F "glue_gun" & F "needle" & F "scissors"', "[start_time, end_time]",
                           detector, StageFrameValidator, PipelineConfig(progress_bar=False))
    assert out.segment_mode == "union"
    segments = _sec(out.segments)
    assert len(segments) == 3
    for t in (10, 60, 110):
        assert any(s <= t <= e for s, e in segments), (t, segments)


def test_target_window_padding():
    detector = ScriptedDetector(_schedule({"a": [(30, 44)], "b": [(45, 50)]}))
    plain = run_localization(_video(120), ["a", "b"], '"a" U "b"', "[start_time, end_time]",
                             detector, StageFrameValidator, PipelineConfig(progress_bar=False))
    padded = run_localization(_video(120), ["a", "b"], '"a" U "b"', "[start_time - 5, end_time + 2.5]",
                              detector, StageFrameValidator, PipelineConfig(progress_bar=False))
    (s0, e0), = plain.segments
    (s1, e1), = padded.segments
    assert s1 == s0 - 5 * FPS and e1 == int(e0 + 2.5 * FPS)


def test_nothing_detected_falls_back():
    out = run_localization(_video(30), ["a", "b"], '"a" U "b"', "[start_time, end_time]",
                           ScriptedDetector({}), StageFrameValidator, PipelineConfig(progress_bar=False))
    assert out.foi and out.segments  # falls back to the sampled frames instead of crashing


def test_single_mode_nothing_detected():
    out = run_localization(_video(30), ["a", "b"], '"a" U "b"', "[start_time, end_time]",
                           ScriptedDetector({}), FrameValidator,
                           PipelineConfig(segment_mode="single", bridge_seconds=10, context_seconds=10,
                                          progress_bar=False))
    assert out.foi == [-1] and out.segments == [-1]


# ---------------------------------------------------------------- NeuS-QA regression vs. the old code

@pytest.fixture(scope="module")
def old_ablation_module(tmp_path_factory):
    """nsvs_ablation.run_nsvs_ablation from the last pre-restructure commit (the code behind the reported NeuS-QA runs)."""
    root = tmp_path_factory.mktemp("old")
    try:
        archive = subprocess.run(["git", "-C", OLD_CODE_REPO, "archive", OLD_CODE_COMMIT, "nsvqa"], check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"set NEUSQA_REPO to a NeuS-QA clone containing commit {OLD_CODE_COMMIT}")
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    stubs = {"nsvqa.nsvs.vlm.vllm_client": types.SimpleNamespace(VLLMClient=None),
             "decord": types.SimpleNamespace(VideoReader=None, cpu=None)}
    added = [name for name in stubs if name not in sys.modules]
    for name in added:
        sys.modules[name] = stubs[name]
    sys.path.insert(0, str(root))
    try:
        from nsvqa.nsvs import nsvs_ablation
        yield nsvs_ablation
    finally:
        sys.path.remove(str(root))
        for name in [m for m in sys.modules if m == "nsvqa" or m.startswith("nsvqa.")] + added:
            sys.modules.pop(name, None)


def _random_case(seed):
    rng = random.Random(seed)
    props = ["a", "b", "c"]
    spans = {p: [(s, s + rng.randint(3, 20)) for s in sorted(rng.sample(range(0, 160), rng.randint(1, 5)))]
             for p in props}
    spec = rng.choice(['"a" U "b"', '("a" & "b") U "c"', '"a" & "b"', '"a" U ("b" | "c")'])
    window = rng.choice(["[start_time, end_time]", "[start_time - 10, end_time + 5]", "[start_time, end_time + 20]"])
    return props, spec, window, _schedule(spans)


@pytest.mark.parametrize("seed", range(40))
def test_neusqa_matches_old_ablation_path(old_ablation_module, seed):
    props, spec, window, schedule = _random_case(seed)
    detector = ScriptedDetector(schedule)
    video = _video(180)

    old_config = types.SimpleNamespace(vlm_method="sequential", read_video_method="uniform",
                                       return_segments=False, adaptive_gt=False)
    old_foi, old_det, old_merged, _ = old_ablation_module.run_nsvs_ablation(
        video, "v.mp4", props, spec, window, model="m", device=0, vlm=detector,
        measure_metrics=True, config=old_config)

    from neusqa.pipeline import default_config
    new = run_localization(video, props, spec, window, detector, FrameValidator,
                           default_config(progress_bar=False))

    assert new.foi == ([old_foi] if old_foi != [-1] and not isinstance(old_foi[0], list) else old_foi)
    assert [sorted(s) for s in new.detections] == [sorted(s) for s in old_det]
    assert new.segments == old_merged


# ---------------------------------------------------------------- presets and configs

class _Preset(LocalizationPipeline):
    frame_validator_cls = StageFrameValidator

    def build_detector(self):
        raise AssertionError("detector is injected")

    def sample_frames(self, video_path, propositions):
        return _video(120)


def test_localize_with_cached_llm_outputs():
    detector = ScriptedDetector(_schedule({"a": [(30, 44)], "b": [(45, 50)]}))
    pipe = _Preset(PipelineConfig(progress_bar=False), detector=detector)
    result = pipe.localize(
        "v.mp4", "what happens after a?", ["x", "y"], sample_id="q1",
        puls={"proposition": ["a", "b"], "specification": '"a" U "b"', "conversation_history": ""},
        target_identification={"frame_window": "[start_time, end_time]", "explanation": "", "conversation_history": ""},
        measure_metrics=True,
    )
    assert result.error is None and result.segment_mode == "intersect"
    entry = result.to_entry()
    assert entry["frames_of_interest"] == result.segments
    assert entry["metadata"] == {"fps": FPS, "frame_count": 120 * FPS}
    assert {"PULS_time", "NeuS_time", "num_frame_windows", "model_checks_time"} <= set(result.metrics)
    json.dumps(entry)  # must serialize


def test_invalid_puls_short_circuits():
    pipe = _Preset(PipelineConfig(progress_bar=False), detector=ScriptedDetector({}))
    result = pipe.localize("v.mp4", "q", None, puls={"proposition": [], "specification": "", "is_valid": False},
                           measure_metrics=True)
    assert not result.is_valid and result.error
    entry = result.to_entry()
    assert entry["frames_of_interest"] == [-1] and entry["nsvs"]["output"] == [-1]
    assert entry["metadata"]["error"] == result.error
    assert {"PULS_time", "completion_time"} <= set(result.metrics)


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(REPO, "configs", "*.json"))))
def test_pipeline_configs_load(path):
    data = json.load(open(path))
    config = PipelineConfig.from_dict(data)
    assert config.segment_mode in ("single", "auto")


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(REPO, "configs", "ablation", "*.json"))))
def test_ablation_configs_load(path):
    from rtneus.ablation.pipeline import load_ablation_config

    data = json.load(open(path))
    choice, config = load_ablation_config(data)
    assert choice.vlm_method in ("sequential", "batched")
    assert config.segment_mode == ("intersect" if data["return_segments"] else "single")
    assert config.bridge_seconds == 10 and config.context_seconds == 10
    assert (config.record_high_confidence == 0.8) == bool(data.get("adaptive_gt"))


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="segmnet_mode"):
        PipelineConfig.from_dict({"segmnet_mode": "union"})


def test_cli_arguments():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    try:
        import localize
    finally:
        sys.path.pop(0)
    args = localize.parse_args(["--pipeline", "rtneus", "--config", "configs/rtneus_lvb.json",
                                "--output_dir", "out", "--port_number", "3"])
    assert args.device == 3 and args.benchmark == "lvb"
    with pytest.raises(SystemExit):
        localize.parse_args(["--pipeline", "ablation", "--output_dir", "out"])


@pytest.mark.parametrize("spec,props,expected", [
    ("person opens the DOOR UNTIL person enters ORANGE room", ["person opens the DOOR", "person enters ORANGE room"],
     '"person_opens_the_door" U "person_enters_orange_room"'),
    ("subtitle_'ROCK AND ROLL NOT DEAD' UNTIL crowd cheers", ["subtitle_'ROCK AND ROLL NOT DEAD'", "crowd cheers"],
     '"subtitle_rock_and_roll_not_dead" U "crowd_cheers"'),
    ("robots stand silently UNTIL (red robot starts blinking OR NOT green robot turns off)",
     ["robots stand silently", "red robot starts blinking", "green robot turns off"],
     '"robots_stand_silently" U ( "red_robot_starts_blinking" | ! "green_robot_turns_off" )'),
    ("ALWAYS car is red AND NOT dog barks", ["car is red", "dog barks"], 'G "car_is_red" & ! "dog_barks"'),
])
def test_puls_keywords_inside_propositions(spec, props, expected):
    pytest.importorskip("openai")
    from neus_core.puls.puls import process_specification

    assert process_specification(spec, props)[1] == expected
