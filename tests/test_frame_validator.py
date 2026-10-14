"""Frame validator behaviour: upstream NeuS-QA (frozen) vs RT-NeuS stage-wise.

Run: uv run --no-project --with pytest --with numpy --with opencv-python-headless pytest tests
"""

import pytest

from neusqa.frame_validator import FrameValidator
from rtneus.frame_validator import StageFrameValidator


class _Obj:
    def __init__(self, probability):
        self.probability = probability

    def get_detected_probability(self):
        return self.probability


class _Frame:
    def __init__(self, props, detected):
        self.object_of_interest = {p: _Obj(1.0 if p in detected else 0.0) for p in props}

    def thresholded_detected_objects(self, threshold):
        return {
            p: o.probability for p, o in self.object_of_interest.items() if o.probability > threshold
        }


PROPS = ["a", "b", "c", "d"]


def _validate(cls, spec, detected):
    validator = cls(f"P>=0.80 [ {spec} ]", threshold_of_probability=0.5)
    return validator.validate_frame(_Frame(PROPS, detected))


# (spec, detected, upstream result, correct result)
CASES = [
    ('("a" & "b") U "c"', {"a"}, False, False),
    ('("a" & "b") U "c"', {"a", "b"}, True, True),
    ('("a" & "b") U "c"', {"c"}, True, True),
    ('"a" U "b"', {"a"}, True, True),
    ('"a" U "b"', {"b"}, True, True),
    ('"a" | "b"', {"a"}, False, True),
    ('"a" | "b"', {"b"}, False, True),
    ('"a" | "b"', {"c"}, False, False),
    ('("a" | "b") U "c"', {"a"}, False, True),
    ('("a" | "b") U "c"', {"c"}, False, True),
    ('"a" U ("b" | "c")', {"a"}, False, True),
    ('"a" U ("b" | "c")', {"b"}, False, True),
    ('"a" & !"b"', {"a"}, False, True),
    ('"a" & !"b"', {"a", "b"}, False, False),
    ('!"a" & !"b" & "c"', {"c"}, False, True),
    ('!"a" & !"b" & "c"', {"b", "c"}, True, False),
    ('("a" & !"b") U "b"', {"b"}, False, True),
    ('"a" & "b" & "c"', {"a", "b"}, True, True),  # soft-AND majority vote kept
    ('"a" & "b" & "c"', {"a"}, False, False),
    ('F "a"', {"a"}, True, True),
    ('G "a"', {"a"}, True, True),
    ('"a" U "b"', set(), False, False),
]

# multi-operator (N-stage) formulas, checked against the stage validator only
MULTI_CASES = [
    ('"a" U "b" U "c"', {"a"}, True),
    ('"a" U "b" U "c"', {"c"}, True),
    ('"a" U ("b" U "c")', {"b"}, True),
    ('("a" & "b") U "c" U "d"', {"a"}, False),
    ('("a" & "b") U "c" U "d"', {"a", "b"}, True),
    ('("a" & "b") U "c" U "d"', {"d"}, True),
    ('("a" & "b") | "c"', {"a"}, False),
    ('("a" & "b") | "c"', {"c"}, True),
    ('("a" & !"b") U "b"', {"b"}, True),
    # sequence questions (MLVU / Video-MME): events happen at different times
    ('F "a" & F "b" & F "c"', {"a"}, True),
    ('F "a" & F "b" & F "c"', {"c"}, True),
    ('F "a" & F "b" & F "c"', {"d"}, False),
    # selection questions: any candidate shown
    ('F "a" | F "b" | F "c"', {"b"}, True),
    ('F "a" | F "b" | F "c"', {"d"}, False),
    # present-tense conjunct must still hold; an eventual one may come later
    ('"a" & F "b"', {"a"}, True),
    ('"a" & F "b"', {"b"}, True),
    ('"a" & "b" & F "c"', {"a", "b"}, True),
    ('"a" & "b" & F "c"', {"a"}, False),  # present-tense AND still needs a strict majority
    ('"a" & "b" & F "c"', {"d"}, False),
    # Until nested inside another operator: either side is evidence
    ('("a" U "b") | "c"', {"b"}, True),
    ('("a" U "b") | "c"', {"d"}, False),
]


@pytest.mark.parametrize("spec,detected,upstream,_", CASES)
def test_upstream_validator_unchanged(spec, detected, upstream, _):
    assert _validate(FrameValidator, spec, detected) is upstream


@pytest.mark.parametrize("spec,detected,_,correct", CASES)
def test_stage_validator(spec, detected, _, correct):
    assert _validate(StageFrameValidator, spec, detected) is correct


@pytest.mark.parametrize("spec,detected,correct", MULTI_CASES)
def test_stage_validator_multi_stage(spec, detected, correct):
    assert _validate(StageFrameValidator, spec, detected) is correct


def test_unparseable_spec_falls_back_to_any_prop():
    validator = StageFrameValidator('P>=0.80 [ ("a" & ) U "b" ]')
    assert _validate(StageFrameValidator, '("a" & ) U "b"', {"b"}) is True
    assert validator.stage_of("a") == 0


def test_nested_until_splits_into_stages():
    validator = StageFrameValidator('P>=0.80 [ "a" U ("b" U "c") ]')
    assert [validator.stage_of(p) for p in "abc"] == [0, 1, 2]


def test_stage_of_uses_exact_names():
    validator = StageFrameValidator('P>=0.80 [ "man_runs" U "man" ]')
    assert validator.stage_of("man_runs") == 0
    assert validator.stage_of("man") == 1


def test_stage_of_skips_negated_occurrence():
    validator = StageFrameValidator('P>=0.80 [ ("a" & !"b") U "b" ]')
    assert validator.stage_of("b") == 1
    assert validator.stage_of("missing") is None


def test_stage_of_without_until():
    validator = StageFrameValidator('P>=0.80 [ "a" & !"b" ]')
    assert validator.stage_of("a") == 0
    assert validator.stage_of("b") is None


def test_capital_letters_in_props_are_preserved():
    validator = StageFrameValidator('P>=0.80 [ "subtitle_Go_Forward" U "man_runs" ]')
    assert validator.stage_of("subtitle_Go_Forward") == 0
