"""Ablations: any combination of NeuS-QA / RT-NeuS components on the shared engine.

The legacy ablation configs (``vlm_method``, ``read_video_method``,
``return_segments``, ``adaptive_gt``) load unchanged; every other key is a
``PipelineConfig`` field. Defaults reproduce the settings the ablation runs
used (upstream frame validator, single-chain PULS, 10 s bridge/context).
"""

import json
from dataclasses import dataclass

from neus_core.config import PipelineConfig
from neus_core.pipeline import LocalizationPipeline

VLM_METHODS = ("sequential", "batched", "kv_cache")
READ_VIDEO_METHODS = ("uniform", "adaptive", "stage1", "stage2")
FRAME_VALIDATORS = ("upstream", "stage")


@dataclass
class AblationChoice:
    vlm_method: str = "sequential"
    read_video_method: str = "uniform"
    frame_validator: str = "upstream"

    def __post_init__(self):
        for value, allowed in [(self.vlm_method, VLM_METHODS), (self.read_video_method, READ_VIDEO_METHODS),
                               (self.frame_validator, FRAME_VALIDATORS)]:
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")


def load_ablation_config(data: dict, **overrides) -> tuple[AblationChoice, PipelineConfig]:
    """Split an ablation config into component choices and engine settings."""
    data = {**data, **overrides}
    choice = AblationChoice(
        vlm_method=data.pop("vlm_method", data.pop("vlm_type", "sequential")),
        read_video_method=data.pop("read_video_method", "uniform"),
        frame_validator=data.pop("frame_validator", "upstream"),
    )
    return_segments = data.pop("return_segments", False)
    if data.pop("adaptive_gt", False):
        data.setdefault("record_high_confidence", 0.8)
    data.setdefault("segment_mode", "intersect" if return_segments else "single")
    defaults = dict(puls="single", bridge_seconds=10, context_seconds=10)
    return choice, PipelineConfig.from_dict({**defaults, **data})


class AblationPipeline(LocalizationPipeline):
    def __init__(self, choice: AblationChoice, config: PipelineConfig, **kwargs):
        self.choice = choice
        if choice.frame_validator == "stage":
            from rtneus.frame_validator import StageFrameValidator
            self.frame_validator_cls = StageFrameValidator
        else:
            from neusqa.frame_validator import FrameValidator
            self.frame_validator_cls = FrameValidator
        super().__init__(config, **kwargs)

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "AblationPipeline":
        with open(path) as f:
            choice, config = load_ablation_config(json.load(f))
        return cls(choice, config, **kwargs)

    def build_detector(self):
        args = dict(model_name=self.config.vlm, device=self.device, max_patch=1)
        if self.choice.vlm_method == "sequential":
            from neusqa.vlm import SequentialInternVL
            return SequentialInternVL(**args)
        if self.choice.vlm_method == "batched":
            from rtneus.ablation.vlm_batched import BatchedInternVL
            return BatchedInternVL(**args)
        from rtneus.vlm import KVCacheInternVL
        return KVCacheInternVL(**args)

    def build_clip_model(self):
        if self.choice.read_video_method == "uniform":
            return None
        from rtneus.pipeline import load_clip
        return load_clip(self.config.clip_model, self.device)

    def sample_frames(self, video_path, propositions):
        method = self.choice.read_video_method
        if method == "uniform":
            from neusqa.sampling import read_video_uniform
            return read_video_uniform(None, video_path, propositions, sample_rate=self.config.sample_rate)
        if method == "adaptive":
            from rtneus.sampling import read_video_adaptive
            return read_video_adaptive(model=self.clip_model, video_path=video_path, propositions=propositions,
                                       threshold=self.config.clip_threshold)
        from rtneus.ablation import sampling
        sampler = sampling.read_video_stage1 if method == "stage1" else sampling.read_video_stage2
        return sampler(model=self.clip_model, video_path=video_path, propositions=propositions,
                       threshold=self.config.clip_threshold)
