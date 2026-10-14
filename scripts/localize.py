"""Run NeuS-QA, RT-NeuS or an ablation over one split of a benchmark.

Replaces scripts/lmms_eval/nsvs_only*.py. Output layout (unchanged, so the
lmms-eval preparation scripts keep working):

    <output_dir>/nsvqa_output/nsvqa_output_<split>.json        entries + localization results
    <output_dir>/postprocess_output/postprocess_output_<split>.json   (with --save_crop)
    <output_dir>/llm_conversation_history/                      PULS / target-ID transcripts
    <output_dir>/full_time_metrics.json                         (with --measure_metrics)
    <output_dir>/run_config.json                                exact settings of this run

Example:
    python scripts/localize.py --pipeline rtneus --config configs/rtneus_lvb.json --benchmark lvb \
        --data_dir /data/LongVideoBench --burned_dir /data/LongVideoBench/burn-subtitles/... \
        --output_dir results/rtneus/lvb/split_1 --current_split 1 --total_splits 4 \
        --categories T3E E3E T3O O3O --device 0 --measure_metrics --save_crop
"""

import argparse
import json
import logging
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_pipeline(args, llm_log_dir):
    overrides = {"vlm": args.vlm_model_name} if args.vlm_model_name else {}
    if args.pipeline == "ablation":
        from rtneus.ablation import AblationPipeline, load_ablation_config

        with open(args.config) as f:
            choice, config = load_ablation_config(json.load(f), **overrides)
        pipeline = AblationPipeline(choice, config, device=args.device, llm_log_dir=llm_log_dir)
        run_config = {"pipeline": "ablation", **vars(choice), **config.to_dict()}
        return pipeline, run_config

    if args.pipeline == "neusqa":
        from neusqa import NeuSQA as Pipeline, default_config
    else:
        from rtneus import RTNeuS as Pipeline, default_config
    config = default_config(**overrides)
    if args.config:
        from neus_core.config import PipelineConfig
        config = PipelineConfig.from_dict({**config.to_dict(), **json.load(open(args.config))}, **overrides)
    pipeline = Pipeline(config, device=args.device, llm_log_dir=llm_log_dir)
    return pipeline, {"pipeline": args.pipeline, **config.to_dict()}


def build_data_loader(args, postprocess_path):
    from neus_core.datasets.custom import Custom
    from neus_core.datasets.longvideobench import LongVideoBench
    from neus_core.datasets.mlvu import MLVU
    from neus_core.datasets.video_mme import VideoMME

    if args.benchmark == "lvb":
        return LongVideoBench(dataset_path=args.data_dir, burned_path=args.burned_dir,
                              postprocess_dir=postprocess_path, categories=args.categories)
    if args.benchmark == "mlvu":
        return MLVU(dataset_path=args.data_dir, burned_path=args.burned_dir,
                    postprocess_dir=postprocess_path, categories=args.categories)
    if args.benchmark == "video_mme":
        return VideoMME(dataset_path=args.data_dir, burned_path=args.burned_dir,
                        postprocess_dir=postprocess_path, categories=args.categories)
    if args.benchmark == "custom":
        with open(args.custom_data) as f:
            raw_data = json.load(f)
        return Custom(raw_data=raw_data, burned_dir=args.burned_dir, dataset_path=args.data_dir,
                      postprocess_dir=postprocess_path)
    raise ValueError(f"Unknown benchmark {args.benchmark}")


def main(args):
    experiment_dir = args.output_dir
    for sub in ("nsvqa_output", "postprocess_output"):
        os.makedirs(os.path.join(experiment_dir, sub), exist_ok=True)
    output_path = os.path.join(experiment_dir, "nsvqa_output", f"nsvqa_output_{args.current_split}.json")
    postprocess_path = os.path.join(experiment_dir, "postprocess_output", f"postprocess_output_{args.current_split}.json")
    llm_log_dir = os.path.join(experiment_dir, "llm_conversation_history")

    pipeline, run_config = build_pipeline(args, llm_log_dir)
    with open(os.path.join(experiment_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"RUN CONFIG: {run_config}")

    data_loader = build_data_loader(args, postprocess_path)
    data = data_loader.load_data()
    start = (len(data) * (args.current_split - 1)) // args.total_splits
    end = (len(data) * args.current_split) // args.total_splits
    print(f"Data Loading Complete! {len(data)} entries, running {start}..{end - 1}")

    output, metrics_output = [], []
    for i in range(start, end):
        entry = data[i]
        sample_id = entry["metadata"]["id"]
        print("\n" + "*" * 50 + f" {i}/{len(data) - 1} {sample_id} " + "*" * 50)

        result = pipeline.localize(
            video_path=entry["paths"]["video_path"],
            question=entry["question"],
            candidates=entry.get("candidates"),
            sample_id=sample_id,
            measure_metrics=args.measure_metrics,
        )
        if not result.is_valid:
            # Kept in the output with frames_of_interest [-1] so VQA falls back to the full video
            logging.critical(f"{result.error}, Returning Full Video Length")

        fields = result.to_entry()
        entry.setdefault("metadata", {}).update(fields.pop("metadata"))
        entry.update(fields)
        output.append(entry)

        if args.measure_metrics:
            metrics_output.append({**result.metrics, "metadata": dict(entry["metadata"])})
        print(f"Localization complete for {sample_id}: {result.segments} ({result.segment_mode})")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    if args.measure_metrics:
        with open(os.path.join(experiment_dir, "full_time_metrics.json"), "w") as f:
            json.dump(metrics_output, f, indent=4)
    if args.save_crop:
        data_loader.postprocess_data(output_path, args.measure_metrics)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=["neusqa", "rtneus", "ablation"], required=True)
    parser.add_argument("--config", type=str, help="JSON config (required for --pipeline ablation)")
    parser.add_argument("--benchmark", choices=["lvb", "mlvu", "video_mme", "custom"], default="lvb")
    parser.add_argument("--custom_data", type=str, help="entries JSON for --benchmark custom")
    parser.add_argument("--vlm_model_name", type=str, help="overrides the config's vlm, e.g. InternVL2-8B")
    parser.add_argument("--device", "--port_number", dest="device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--burned_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--current_split", type=int, default=1)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--categories", nargs="+", type=str)
    parser.add_argument("--measure_metrics", action="store_true")
    parser.add_argument("--save_crop", action="store_true", help="crop the localized segments with ffmpeg")
    args = parser.parse_args(argv)
    if args.pipeline == "ablation" and not args.config:
        parser.error("--pipeline ablation requires --config")
    if args.benchmark == "custom" and not args.custom_data:
        parser.error("--benchmark custom requires --custom_data")
    return args


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
