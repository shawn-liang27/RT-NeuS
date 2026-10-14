"""RT-NeuS: CLIP-adaptive sampling, KV-cache batched proposition detection,
stage-wise frame validation and multi-segment output."""

from neus_core.config import PipelineConfig
from neus_core.pipeline import LocalizationPipeline
from rtneus.frame_validator import StageFrameValidator


def default_config(**overrides) -> PipelineConfig:
    """RT-NeuS settings; ``puls="multi"`` for candidate-style benchmarks (MLVU, Video-MME)."""
    return PipelineConfig.from_dict(
        dict(puls="single", segment_mode="auto", bridge_seconds=3, context_seconds=3), **overrides
    )


def load_clip(name: str, device: int):
    from sentence_transformers import SentenceTransformer

    clip_model = SentenceTransformer(name, device=f"cuda:{device}")
    clip_model.eval()
    return clip_model


class RTNeuS(LocalizationPipeline):
    frame_validator_cls = StageFrameValidator

    def __init__(self, config: PipelineConfig | None = None, **kwargs):
        super().__init__(config or default_config(), **kwargs)

    def build_detector(self):
        from rtneus.vlm import KVCacheInternVL
        return KVCacheInternVL(model_name=self.config.vlm, device=self.device, max_patch=1)

    def build_clip_model(self):
        return load_clip(self.config.clip_model, self.device)

    def sample_frames(self, video_path, propositions):
        from rtneus.sampling import read_video_adaptive
        return read_video_adaptive(
            model=self.clip_model, video_path=video_path, propositions=propositions,
            threshold=self.config.clip_threshold,
        )
