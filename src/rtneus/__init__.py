"""RT-NeuS: real-time neuro-symbolic video question answering.

    from rtneus import RTNeuS
    result = RTNeuS(device=0).localize("video.mp4", question, candidates)
"""

__all__ = ["RTNeuS", "default_config"]


def __getattr__(name):
    # Lazy so that importing a submodule (e.g. rtneus.frame_validator) does not load torch/stormpy
    if name in __all__:
        from rtneus import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'rtneus' has no attribute {name!r}")
