"""scripts/evaluation/prepare_lmms_segments.py: lmms-eval question files with segments, no cropping."""

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "evaluation"))
import prepare_lmms_segments as prep  # noqa: E402


def _write_run(tmp_path, entries, splits=2):
    for s in range(1, splits + 1):
        d = tmp_path / "run" / f"split_{s}" / "nsvqa_output"
        d.mkdir(parents=True)
        (d / f"nsvqa_output_{s}.json").write_text(json.dumps(entries[s - 1::splits]))
    return str(tmp_path / "run")


def _lvb_entry(i, foi):
    return {"question": f"q{i}", "candidates": ["a", "b"], "correct_choice": 0, "frames_of_interest": foi,
            "paths": {"video_path": f"/burned/v{i}.mp4"}, "metadata": {"id": f"id{i}", "video_id": f"v{i}"}}


def test_lvb(tmp_path):
    data = tmp_path / "lvb"
    data.mkdir()
    originals = [{"id": f"id{i}", "video_path": f"v{i}.mp4", "candidates": ["a", "b"], "question": f"q{i}",
                  "correct_choice": 0} for i in range(3)]
    (data / "lvb_val.json").write_text(json.dumps(originals))
    run = _write_run(tmp_path, [_lvb_entry(0, [[13225, 13605]]), _lvb_entry(1, [-1]),
                                _lvb_entry(2, [(100, 200), (300, 400)])])
    out = tmp_path / "lmms" / "neusqa" / "lvb_val.json"
    prep.main(["--benchmark", "lvb", "--run_dir", run, "--data_dir", str(data), "--out", str(out)])

    docs = {d["id"]: d for d in json.load(open(out))}
    assert docs["id0"]["frames_of_interest"] == [[13225, 13605]]
    assert docs["id1"]["frames_of_interest"] == [[-1]]                 # whole video
    assert docs["id2"]["frames_of_interest"] == [[100, 200], [300, 400]]
    assert docs["id0"]["video_path"] == "/burned/v0.mp4"               # the video RT-NeuS localized on
    assert docs["id0"]["option0"] == "a" and docs["id0"]["option1"] == "b"
    assert not list((tmp_path / "lmms" / "neusqa").glob("*.mp4"))      # nothing cropped or copied


def test_full_video_baseline(tmp_path):
    data = tmp_path / "lvb"
    data.mkdir()
    (data / "lvb_val.json").write_text(json.dumps([{"id": "id0", "video_path": "v0.mp4", "candidates": ["a"]}]))
    run = _write_run(tmp_path, [_lvb_entry(0, [[1, 2]])], splits=1)
    out = tmp_path / "original.json"
    prep.main(["--benchmark", "lvb", "--run_dir", run, "--data_dir", str(data), "--out", str(out), "--full_video"])
    assert json.load(open(out))[0]["frames_of_interest"] == [[-1]]


def test_video_mme(tmp_path):
    data = tmp_path / "vmme"
    data.mkdir()
    (data / "dataset.json").write_text(json.dumps([
        {"question_id": "001-1", "videoID": "abc", "question": "q", "options": ["A. x", "B. y"], "answer": "A",
         "duration": "short", "domain": "d", "sub_category": "s", "task_type": "t"}]))
    entry = {"question": "q", "candidates": ["A. x", "B. y"], "correct_choice": "A", "frames_of_interest": [[350, 650]],
             "paths": {"video_path": "/burned/short/abc.mp4"}, "metadata": {"id": "abc_001-1", "question_id": "001-1"}}
    run = _write_run(tmp_path, [entry], splits=1)
    out = tmp_path / "experiment_1.json"
    prep.main(["--benchmark", "video_mme", "--run_dir", run, "--data_dir", str(data), "--out", str(out)])
    (doc,) = json.load(open(out))
    assert doc["videoID"] == "abc" and doc["options"] == ["A. x", "B. y"]
    assert doc["frames_of_interest"] == [[350, 650]]


def test_mlvu_matches_old_prepare_script(tmp_path):
    entry = {"question": "q", "candidates": ["x", "y"], "correct_choice": "x", "frames_of_interest": [-1],
             "paths": {"video_path": "/mlvu/video/2_needle/n1.mp4"},
             "metadata": {"video_id": "n1.mp4", "duration": 300, "question_type": "findNeedle", "id": "n1"}}
    run = _write_run(tmp_path, [entry], splits=1)
    out = tmp_path / "experiment_1.json"
    prep.main(["--benchmark", "mlvu", "--run_dir", run, "--out", str(out)])
    assert json.load(open(out)) == [{
        "video": "n1.mp4", "video_path": "/mlvu/video/2_needle/n1.mp4", "duration": 300, "question": "q",
        "candidates": ["x", "y"], "answer": "x", "question_type": "findNeedle", "frames_of_interest": [[-1]]}]


def test_requires_data_dir(tmp_path):
    with pytest.raises(SystemExit):
        prep.main(["--benchmark", "lvb", "--run_dir", str(tmp_path), "--out", str(tmp_path / "x.json")])
