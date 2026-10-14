"""NeuS-QA: neuro-symbolic video question answering (baseline).

    from neusqa import NeuSQA
    result = NeuSQA(device=0).localize("video.mp4", question, candidates)
"""

__all__ = ["NeuSQA", "default_config"]


def __getattr__(name):
    # Lazy so that importing a submodule (e.g. neusqa.frame_validator) does not load torch/stormpy
    if name in __all__:
        from neusqa import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'neusqa' has no attribute {name!r}")
