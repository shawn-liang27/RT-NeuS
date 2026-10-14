"""Build an lmms-eval question file that carries RT-NeuS segments instead of cropped videos.

Reads every split's ``nsvqa_output_*.json`` from a localization run, attaches each question's
``frames_of_interest`` (frame indices of the video RT-NeuS localized on; ``[-1]`` -> ``[[-1]]`` =
whole video) to the benchmark's original question entry, and writes the file the patched lmms-eval
task reads (third_party/lmms-eval). No video is cropped, copied or re-encoded: lmms-eval opens the
same video file RT-NeuS used and samples frames only inside the segments.

    python scripts/evaluation/prepare_lmms_segments.py --benchmark lvb \
        --run_dir results/rtneus/lvb/run_1 --data_dir "$LVB_DIR" --out results/rtneus/lvb/run_1/lmms/neusqa/lvb_val.json

``--run_dir`` holds ``split_*/nsvqa_output/nsvqa_output_*.json`` (or is a single split folder).
``--full_video`` writes the same questions with every ``frames_of_interest`` set to ``[[-1]]``: the
"original" full-video comparison, also without copying any video.
"""

import argparse
import glob
import json
import os
import sys


def load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "**", "nsvqa_output_*.json"), recursive=True))
    if not files:
        sys.exit(f"No nsvqa_output_*.json under {run_dir}")
    entries = [entry for path in files for entry in json.load(open(path))]
    print(f"Loaded {len(entries)} localized questions from {len(files)} file(s)")
    return entries


def segments_of(entry):
    """``frames_of_interest`` as the lmms-eval tasks expect it: [[start, end], ...] or [[-1]]."""
    foi = entry.get("frames_of_interest")
    if not foi or foi == [-1] or foi == [[-1]]:
        return [[-1]]
    return [[int(start), int(end)] for start, end in foi]


def prepare_lvb(entries, data_dir):
    """LongVideoBench: lvb_val.json entries (+ option<i> fields) with the localized video's absolute path."""
    originals = {item["id"]: item for item in json.load(open(os.path.join(data_dir, "lvb_val.json")))}
    out = []
    for entry in entries:
        original = originals.get(entry["metadata"]["id"])
        if original is None:
            print(f"[WARNING] {entry['metadata']['id']} not in lvb_val.json, skipped")
            continue
        doc = dict(original)
        # the burned video RT-NeuS localized on; an absolute path makes the task ignore its cache_dir
        doc["video_path"] = os.path.abspath(entry["paths"]["video_path"])
        for i, candidate in enumerate(doc["candidates"]):
            doc[f"option{i}"] = candidate
        doc["frames_of_interest"] = segments_of(entry)
        out.append(doc)
    return out


def prepare_video_mme(entries, data_dir):
    """Video-MME: the HF dataset.json entries (one per question) with frames_of_interest."""
    originals = {item["question_id"]: item for item in json.load(open(os.path.join(data_dir, "dataset.json")))}
    out = []
    for entry in entries:
        original = originals.get(entry["metadata"]["question_id"])
        if original is None:
            print(f"[WARNING] {entry['metadata']['question_id']} not in dataset.json, skipped")
            continue
        doc = dict(original)
        doc["frames_of_interest"] = segments_of(entry)
        out.append(doc)
    return out


def prepare_mlvu(entries, data_dir=None):
    """MLVU: the fields tasks/mlvu/utils.py reads (same as prepare_lmms_mlvu.py)."""
    return [
        {
            "video": entry["metadata"]["video_id"],
            "video_path": entry["paths"]["video_path"],
            "duration": entry["metadata"]["duration"],
            "question": entry["question"],
            "candidates": entry["candidates"],
            "answer": entry["correct_choice"],
            "question_type": entry["metadata"]["question_type"],
            "frames_of_interest": segments_of(entry),
        }
        for entry in entries
    ]


PREPARE = {"lvb": prepare_lvb, "video_mme": prepare_video_mme, "mlvu": prepare_mlvu}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=sorted(PREPARE), required=True)
    parser.add_argument("--run_dir", required=True, help="localization run (split_* folders or one split)")
    parser.add_argument("--data_dir", help="benchmark data dir: lvb_val.json (lvb) / dataset.json (video_mme)")
    parser.add_argument("--out", required=True, help="question file to write")
    parser.add_argument("--full_video", action="store_true", help="ignore segments: full-video baseline")
    args = parser.parse_args(argv)
    if args.benchmark in ("lvb", "video_mme") and not args.data_dir:
        parser.error(f"--data_dir is required for {args.benchmark}")

    docs = PREPARE[args.benchmark](load_run(args.run_dir), args.data_dir)
    if args.full_video:
        for doc in docs:
            doc["frames_of_interest"] = [[-1]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(docs, f, indent=2)
    full = sum(doc["frames_of_interest"] == [[-1]] for doc in docs)
    print(f"Wrote {len(docs)} questions to {args.out} ({len(docs) - full} with segments, {full} full video)")


if __name__ == "__main__":
    main()
