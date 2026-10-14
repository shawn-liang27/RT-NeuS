"""NeuS-QA baseline: uniform sampling, one proposition per VLM call, upstream frame validator,
a single segment where the automaton and the detections agree."""

from neus_core.config import PipelineConfig
from neus_core.pipeline import LocalizationPipeline
from neusqa.frame_validator import FrameValidator


def default_config(**overrides) -> PipelineConfig:
    """Settings of the reported NeuS-QA runs (experiment_results/neusqa/config.json)."""
    return PipelineConfig.from_dict(
        dict(puls="single", segment_mode="single", bridge_seconds=10, context_seconds=10), **overrides
    )


class NeuSQA(LocalizationPipeline):
    frame_validator_cls = FrameValidator

    def __init__(self, config: PipelineConfig | None = None, **kwargs):
        super().__init__(config or default_config(), **kwargs)

    def build_detector(self):
        from neusqa.vlm import SequentialInternVL
        return SequentialInternVL(model_name=self.config.vlm, device=self.device, max_patch=1)

    def sample_frames(self, video_path, propositions):
        from neusqa.sampling import read_video_uniform
        return read_video_uniform(None, video_path, propositions, sample_rate=self.config.sample_rate)
