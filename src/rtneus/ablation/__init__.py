"""Component ablations for RT-NeuS, built on the same modules as the main pipelines."""

__all__ = ["AblationChoice", "AblationPipeline", "load_ablation_config"]


def __getattr__(name):
    if name in __all__:
        from rtneus.ablation import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'rtneus.ablation' has no attribute {name!r}")
