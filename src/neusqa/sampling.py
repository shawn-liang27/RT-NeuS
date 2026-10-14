"""NeuS-QA frame sampling: a fixed number of frames per second, no CLIP filtering."""

from decord import VideoReader, cpu


def read_video_uniform(model, video_path, propositions,
                       sample_rate=1, threshold=0.23):
    """
    Ablation Baseline:
    1. Uniform Sampling: Pick frames strictly at a constant sample_rate.
    2. No CLIP embeddings or deduplication performed.
    """
    print('[DEBUG] Using Naive Uniform Sampling')
    vr = VideoReader(video_path, ctx=cpu(0))
    fps, frame_count = vr.get_avg_fps(), len(vr)

    # --- 1. STRICT UNIFORM SAMPLING ---
    # Calculate step based on frames per second and desired sample rate
    step = int(max(1, fps / sample_rate))
    uniform_indices = list(range(0, frame_count - 1, step))

    # Efficiently load frames using Decord's batch getter
    final_frames = vr.get_batch(uniform_indices).asnumpy()
    
    num_frames = len(uniform_indices)

    # --- 2. METRICS & RETURN ---
    # Maintain identical structure for comparability in ablation scripts
    clip_metrics = {
        "expected_uniform_count": num_frames,
        "initial_num_frames_sampled" : num_frames,
        "num_frames_retained": num_frames, # No reduction in baseline
        "pct_from_uniform": 1.0 
    }
    print(f'[DEBUG] Using Naive Uniform Sampling\nSampling Complete! Total Length {num_frames}')
    return {
        "images": [f for f in final_frames],
        "original_indices": uniform_indices,
        "video_info": {"fps": fps, "frame_count": frame_count, "sample_rate": sample_rate},
        "video_sample_method": "baseline-uniform",
        "clip_metrics": clip_metrics,
        "final_num_frames_sampled": num_frames,
        "pct_final_num_from_uniform": 1.0
    }