"""Pipeline configuration shared by NeuS-QA, RT-NeuS and the ablations."""

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class PipelineConfig:
    # --- models ---
    vlm: str = "InternVL2-8B"
    clip_model: str = "clip-ViT-B-32"
    puls: str = "single"              # "single" (one Until chain) | "multi" (multi-operator, uses answer candidates)

    # --- frame sampling ---
    sample_rate: float = 1            # frames per second for uniform sampling
    clip_threshold: float = 0.22      # CLIP relevance threshold for adaptive sampling

    # --- detection / model checking ---
    num_of_frame_in_sequence: int = 3
    vlm_detection_threshold: float = 0.349
    detection_threshold: float = 0.5
    tl_satisfaction_threshold: float = 0.6
    model_type: str = "dtmc"
    event_gap_seconds: float = 8      # sampled frames further apart start a new event

    # --- segment output ---
    segment_mode: str = "auto"        # "single" | "intersect" | "union" | "auto"
    bridge_seconds: float = 3         # merge segments closer than this
    context_seconds: float = 3        # context kept around a P->Q handover
    handover_tolerance_seconds: float = 8

    # --- misc ---
    record_high_confidence: float | None = None   # ablation: record windows with any detection above this
    progress_bar: bool = True

    @classmethod
    def from_dict(cls, data: dict, **overrides) -> "PipelineConfig":
        known = {f.name for f in fields(cls)}
        merged = {**data, **overrides}
        unknown = set(merged) - known - {"pipeline", "comment"}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in merged.items() if k in known})

    @classmethod
    def from_json(cls, path: str, **overrides) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f), **overrides)

    def to_dict(self) -> dict:
        return asdict(self)
