"""Shared localization engine for NeuS-QA, RT-NeuS and their ablations.

``run_localization`` is the neuro-symbolic core: split sampled frames into
windows, detect every proposition per window, keep windows the frame validator
accepts, feed them to the video automaton, model-check after each window and
turn the result into frames of interest.

``LocalizationPipeline`` wraps it with the LLM stages (PULS, target
identification) and frame sampling. ``neusqa.NeuSQA``, ``rtneus.RTNeuS`` and
``rtneus.ablation.AblationPipeline`` are presets that choose the components.
"""

import logging
import os
import tempfile
import time
import traceback
from dataclasses import dataclass, field

import tqdm

from neus_core.config import PipelineConfig
from neus_core.model_checker.property_checker import PropertyChecker
from neus_core.model_checker.video_automaton import VideoAutomaton
from neus_core.segments import compute_segments, resolve_segment_mode
from neus_core.video.frames_of_interest import FramesofInterest
from neus_core.video.video_frame import VideoFrame


# ---------------------------------------------------------------- engine

@dataclass
class LocalizationOutput:
    foi: list                 # raw frames of interest, [-1] when nothing was found
    detections: list          # per temporal stage, frame indices where its propositions were detected
    segments: list            # padded + merged (start_frame, end_frame) ranges, [-1] for the full video
    segment_mode: str         # the mode actually used (``auto`` resolved)
    metrics: dict = field(default_factory=dict)


def group_into_continuous_events(frames_with_indices, gap_threshold):
    """Groups (frame, index) tuples into 'event clusters' based on index gaps."""
    if not frames_with_indices:
        return []
    events = []
    current_event = [frames_with_indices[0]]
    for prev, curr in zip(frames_with_indices, frames_with_indices[1:]):
        if curr[1] - prev[1] > gap_threshold:
            events.append(current_event)
            current_event = []
        current_event.append(curr)
    events.append(current_event)
    return events


def run_localization(
    video_data: dict,
    proposition: list,
    specification: str,
    target_window: str,
    detector,
    frame_validator_cls,
    config: PipelineConfig,
    measure_metrics: bool = False,
) -> LocalizationOutput:
    """Find the frames of a sampled video that satisfy ``specification``.

    Args:
        video_data: sampler output with "images", "original_indices" and
            "video_info" ({"fps", "frame_count", "sample_rate"}).
        detector: object with ``batch_detect(frames, propositions, threshold)``.
        frame_validator_cls: frame-level filter class (upstream or stage-wise).
    """
    metrics = {}

    def log_metric(key, value, is_list=False):
        if not measure_metrics:
            return
        if is_list:
            metrics.setdefault(key, []).append(value)
        else:
            metrics[key] = value

    setup_start = time.perf_counter()
    automaton = VideoAutomaton(include_initial_state=True)
    automaton.set_up(proposition_set=proposition)
    checker = PropertyChecker(
        proposition=proposition,
        specification=specification,
        model_type=config.model_type,
        tl_satisfaction_threshold=config.tl_satisfaction_threshold,
        detection_threshold=config.detection_threshold,
        frame_validator_cls=frame_validator_cls,
    )
    log_metric("automaton_set_up_time", time.perf_counter() - setup_start)

    fps = video_data["video_info"]["fps"]
    frames_with_indices = list(zip(video_data["images"], video_data["original_indices"]))
    event_clusters = group_into_continuous_events(frames_with_indices, gap_threshold=config.event_gap_seconds * fps)

    # Non-overlapping windows of num_of_frame_in_sequence frames inside each continuous event
    n = config.num_of_frame_in_sequence
    frame_windows = [event[i:i + n] for event in event_clusters for i in range(0, len(event), n)]
    log_metric("num_frame_windows", len(frame_windows))

    frame_of_interest = FramesofInterest()
    all_detections = [set() for _ in range(checker.num_stages)]
    high_confidence_frames = set()
    yes_no_masses = []
    model_check_count = 0

    for window in tqdm.tqdm(frame_windows, disable=not config.progress_bar):
        current_frames = [f[0] for f in window]
        current_indices = [f[1] for f in window]

        detection_start = time.perf_counter()
        detected_objects = detector.batch_detect(
            seq_of_frames=current_frames,
            scene_descriptions=proposition,
            threshold=config.vlm_detection_threshold,
        )
        frame = VideoFrame(
            frame_idx_list=current_indices,
            object_of_interest={obj.name: obj for obj in detected_objects},
        )
        log_metric("per_frame_window_detection_time", time.perf_counter() - detection_start, True)
        yes_no_masses.extend(obj.yes_no_mass for obj in detected_objects if getattr(obj, "yes_no_mass", None) is not None)

        if config.record_high_confidence is not None:
            if frame.thresholded_detected_objects(config.record_high_confidence):
                high_confidence_frames.update(frame.frame_idx_list)

        if not checker.validate_frame(frame_of_interest=frame):
            continue

        for prop in frame.thresholded_detected_objects(threshold=config.detection_threshold):
            stage = checker.check_split(prop)
            if stage is None:  # prop only appears negated
                continue
            all_detections[min(stage, len(all_detections) - 1)].update(frame.frame_idx_list)

        model_check_start = time.perf_counter()
        automaton.add_frame(frame=frame)
        frame_of_interest.add_frame(frame)
        model_check_count += 1
        if checker.check_automaton(automaton=automaton):
            automaton.reset()
            frame_of_interest.flush_frame_buffer()
        log_metric("model_checks_time", time.perf_counter() - model_check_start, True)

    automaton_foi = frame_of_interest.compile_foi()
    mode = resolve_segment_mode(config.segment_mode, checker.num_stages)
    foi, segments = compute_segments(
        mode=mode,
        video_info=video_data["video_info"],
        all_detections=all_detections,
        automaton_foi=automaton_foi,
        original_indices=video_data.get("original_indices", []),
        target_window=target_window,
        bridge_seconds=config.bridge_seconds,
        context_seconds=config.context_seconds,
        tolerance_seconds=config.handover_tolerance_seconds,
    )
    logging.info(f"Automaton_foi {automaton_foi}\nAll Detections: {all_detections}\n"
                 f"Detected frames of interest ({mode}): {foi}\nMerged Frames of Interests: {segments}")

    log_metric("num_model_checks", model_check_count)
    if yes_no_masses:
        # how well the VLM followed the Yes/No answer rule (mass = P(Yes) + P(No))
        log_metric("yes_no_mass", {
            "count": len(yes_no_masses),
            "min": min(yes_no_masses),
            "mean": sum(yes_no_masses) / len(yes_no_masses),
            "frac_below_0.9": sum(m < 0.9 for m in yes_no_masses) / len(yes_no_masses),
            "frac_below_0.5": sum(m < 0.5 for m in yes_no_masses) / len(yes_no_masses),
        })
    if config.record_high_confidence is not None:
        log_metric("gt_frames", sorted(high_confidence_frames))
    return LocalizationOutput(foi=foi, detections=all_detections, segments=segments, segment_mode=mode, metrics=metrics)


# ---------------------------------------------------------------- full pipeline

@dataclass
class LocalizationResult:
    """Everything one ``localize`` call produced; ``to_entry`` gives the saved JSON fields."""
    puls: dict
    target_identification: dict = field(default_factory=dict)
    foi: list = field(default_factory=lambda: [-1])
    detections: list = field(default_factory=list)
    segments: list = field(default_factory=lambda: [-1])
    segment_mode: str = ""
    video_info: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return bool(self.puls.get("is_valid", bool(self.puls.get("proposition"))))

    def to_entry(self) -> dict:
        """Fields written into each benchmark entry (same layout as the old nsvs_only scripts)."""
        entry = {
            "puls": self.puls,
            "target_identification": self.target_identification,
            "nsvs": {"output": self.foi, "indices": [sorted(s) for s in self.detections],
                     "segment_mode": self.segment_mode},
            "frames_of_interest": self.segments,
        }
        metadata = {}
        if self.video_info:
            metadata["fps"] = self.video_info["fps"]
            metadata["frame_count"] = self.video_info["frame_count"]
        if self.error:
            metadata["error"] = self.error
        entry["metadata"] = metadata
        return entry


class LocalizationPipeline:
    """PULS -> target identification -> frame sampling -> ``run_localization``.

    Subclasses set ``frame_validator_cls`` and implement ``build_detector`` /
    ``sample_frames``. Expensive models are built once in ``__init__``; pass
    ``detector=`` / ``clip_model=`` to reuse already-loaded ones.
    """

    frame_validator_cls = None

    def __init__(self, config: PipelineConfig, device: int = 0, detector=None, clip_model=None,
                 llm_log_dir: str | None = None):
        self.config = config
        self.device = device
        self.llm_log_dir = llm_log_dir or tempfile.mkdtemp(prefix="neus_llm_")
        self.detector = detector if detector is not None else self.build_detector()
        self.clip_model = clip_model if clip_model is not None else self.build_clip_model()

    # --- components (overridden by presets) ---
    def build_detector(self):
        raise NotImplementedError

    def build_clip_model(self):
        """CLIP for frame sampling; ``None`` when the sampler does not use it."""
        return None

    def sample_frames(self, video_path: str, propositions: list) -> dict:
        raise NotImplementedError

    # --- LLM stages ---
    def run_puls(self, question: str, candidates: list | None, sample_id: str) -> dict:
        if self.config.puls == "multi":
            from neus_core.puls_multi.puls import PULS
            output = PULS(question, sample_id, save_dir=self.llm_log_dir, candidates=candidates)
        else:
            from neus_core.puls.puls import PULS
            output = PULS(question, sample_id, save_dir=self.llm_log_dir)
        puls = {
            "proposition": output["proposition"],
            "specification": output["specification"],
            "conversation_history": os.path.abspath(output["saved_path"]),
        }
        if "is_valid" in output:
            puls["is_valid"] = output["is_valid"]
            puls["raw_output"] = output["raw_output"]
        return puls

    def run_target_identification(self, question: str, candidates: list | None, puls: dict, sample_id: str) -> dict:
        from neus_core.target_identification import identify_target
        output = identify_target(
            question, candidates, puls["specification"], puls["conversation_history"], sample_id, self.llm_log_dir
        )
        return {
            "frame_window": output["frame_window"],
            "explanation": output["explanation"],
            "conversation_history": os.path.abspath(output["saved_path"]),
        }

    # --- entry point ---
    def localize(self, video_path: str, question: str, candidates: list | None = None, sample_id: str = "sample",
                 puls: dict | None = None, target_identification: dict | None = None,
                 measure_metrics: bool = False) -> LocalizationResult:
        """Localize the frames needed to answer ``question`` about ``video_path``.

        ``puls`` / ``target_identification`` can be passed from a previous run to
        skip the LLM calls.
        """
        metrics = {}
        start = time.perf_counter()

        step = time.perf_counter()
        puls = puls or self.run_puls(question, candidates, sample_id)
        metrics["PULS_time"] = time.perf_counter() - step
        metrics["num_propositions"] = len(puls["proposition"])
        result = LocalizationResult(puls=puls)
        if not result.is_valid:
            result.error = f"PULS failed to process Propositions and Specification for Task: {sample_id}"
            logging.critical(result.error)
            metrics["frames_of_interest"] = result.segments
            metrics["completion_time"] = time.perf_counter() - start
            if measure_metrics:
                result.metrics = metrics
            return result

        step = time.perf_counter()
        result.target_identification = target_identification or self.run_target_identification(
            question, candidates, puls, sample_id)
        metrics["Target_ID_time"] = time.perf_counter() - step

        step = time.perf_counter()
        video_data = self.sample_frames(video_path, puls["proposition"])
        metrics["video_IO_time"] = time.perf_counter() - step
        metrics.update(video_data.get("clip_metrics", {}))
        for key in ("final_num_frames_sampled", "video_sample_method", "pct_final_num_from_uniform"):
            if key in video_data:
                metrics[key] = video_data[key]
        result.video_info = video_data["video_info"]

        step = time.perf_counter()
        try:
            output = run_localization(
                video_data,
                puls["proposition"],
                puls["specification"],
                result.target_identification["frame_window"],
                detector=self.detector,
                frame_validator_cls=self.frame_validator_cls,
                config=self.config,
                measure_metrics=measure_metrics,
            )
            result.foi, result.detections, result.segments = output.foi, output.detections, output.segments
            result.segment_mode = output.segment_mode
            metrics.update(output.metrics)
        except Exception as e:
            result.error = repr(e)
            logging.error(f"Localization failed for {sample_id}: {e}")
            traceback.print_exc()
        metrics["NeuS_time"] = time.perf_counter() - step
        metrics["frames_of_interest"] = result.segments
        metrics["completion_time"] = time.perf_counter() - start

        if measure_metrics:
            result.metrics = metrics
        if hasattr(self.detector, "clear_gpu_memory"):
            self.detector.clear_gpu_memory()
        return result
