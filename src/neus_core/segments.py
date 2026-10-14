import bisect
import re

def resolve_target_window(target_window):
    inner = target_window.strip()[1:-1]
    parts = inner.split(',')
    result = []
    for part in parts:
        part = part.strip()
        match = re.search(r'([+-])\s*(\d+(?:\.\d+)?)', part)
        if match:
            sign, num = match.groups()
            result.append(float(sign + num))
        else:
            result.append(0)
    return result

# def group_with_gaps(nums, max_gaps=2):
#     if not nums:
#         return []

#     groups = []
#     current_group = [nums[0]]

#     for i in range(1, len(nums)):
#         diff = nums[i] - nums[i-1]
        
#         if diff <= max_gaps + 1:
#             if diff > 1:
#                 current_group.extend(range(nums[i-1] + 1, nums[i] + 1))
#             else:
#                 current_group.append(nums[i])
#         else:
#             groups.append(current_group)
#             current_group = [nums[i]]

#     groups.append(current_group)
#     return groups

def group_with_gaps(nums, max_gaps):
    if not nums:
        return []

    # Sort to ensure we are checking temporal neighbors
    nums = sorted(list(nums))
    
    groups = []
    current_group = [nums[0]]

    for i in range(1, len(nums)):
        # Calculate the jump between this detection and the previous one
        diff = nums[i] - nums[i-1]
        
        # If the jump is smaller than our allowed gap, they belong to the same event
        if diff <= max_gaps:
            current_group.append(nums[i])
        else:
            # Otherwise, finish this group and start a new one
            groups.append(current_group)
            current_group = [nums[i]]

    groups.append(current_group)
    return groups

def intersection_with_gaps(indices, max_gaps=8):
    non_empty = [s for s in indices if s]
    if len(non_empty) == 1:
        return sorted(list(non_empty[0]))

    combined = sorted(list(set().union(*indices)))
    
    if not combined:
        return []

    groups = group_with_gaps(combined, max_gaps)
    largest_set = max(groups, key=lambda g: (max(g) - min(g)))

    return sorted(largest_set)


def densify(indices, max_gaps):
    """Expand sparse indices into continuous ranges by filling gaps within clusters.
    e.g. {100, 103, 106} with max_gaps=5 → {100, 101, 102, 103, 104, 105, 106}
    """
    if not indices:
        return set()
    groups = group_with_gaps(sorted(indices), max_gaps=max_gaps)
    dense = set()
    for g in groups:
        dense.update(range(min(g), max(g) + 1))
    return dense


def find_soft_handover(p_indices, q_indices, tolerance_frames):
    """
    Finds frames where P hands over to Q: a P frame followed by a Q frame no
    more than 'tolerance_frames' later. Order matters for P U Q, so a Q frame
    that precedes every nearby P frame is not a handover.
    """
    handover_points = set()
    p_list = sorted(p_indices)
    q_list = sorted(q_indices)

    if not p_list or not q_list:
        return set()

    for p_val in p_list:
        lo = bisect.bisect_left(q_list, p_val)
        hi = bisect.bisect_right(q_list, p_val + tolerance_frames)
        if lo < hi:
            handover_points.add(p_val)
            handover_points.update(q_list[lo:hi])

    return handover_points

def reconcile_sparse_ltl(video_info, p_indices, q_indices, automaton_foi, target_window, BRIDGE_MULT=3, CONTEXT_SECONDS=5, TOLERANCE=5):

    fps = video_info["fps"]
    sample_rate = video_info.get("sample_rate", 1)
    frame_count = video_info["frame_count"]
    
    MAX_GAPS = BRIDGE_MULT * fps
    WINDOW = CONTEXT_SECONDS * fps      # 300
    tolerance = TOLERANCE * fps

    if q_indices:
        handover_all = find_soft_handover(p_indices, q_indices, tolerance)

        if handover_all:
            handover_sorted = sorted(list(handover_all))
            handover_clusters = group_with_gaps(handover_sorted, max_gaps=MAX_GAPS)
            valid_segments = set()

            for cluster in handover_clusters:
                t_start = min(cluster)
                t_end = max(cluster)

                event_p = {idx for idx in p_indices if (t_start - WINDOW) <= idx <= t_end}
                event_q = {idx for idx in q_indices if t_start <= idx <= (t_end + WINDOW)}

                valid_segments.update(event_p)
                valid_segments.update(event_q)

            # Intersection preserves formal guarantees from the automaton
            # Densify both sets so sparse samples intersect at the range level
            if automaton_foi:
                dense_automaton = densify(automaton_foi, max_gaps=MAX_GAPS)
                dense_segments = densify(valid_segments, max_gaps=MAX_GAPS)
                final_set = dense_segments & dense_automaton
                if not final_set:
                    final_set = dense_automaton
            else:
                final_set = valid_segments
        else:
            final_set = set(automaton_foi)
    else:
        if automaton_foi:
            dense_automaton = densify(automaton_foi, max_gaps=MAX_GAPS)
            dense_props = densify(p_indices, max_gaps=MAX_GAPS)
            final_set = dense_props & dense_automaton
            if not final_set:
                final_set = dense_automaton
        else:
            final_set = p_indices
    if not final_set:
        return [-1]

    final_foi_groups = group_with_gaps(sorted(list(final_set)), max_gaps=MAX_GAPS)

    return final_foi_groups

def reconcile_dynamic_ltl(video_info, all_detections, automaton_foi, target_window_before, target_window_after,
                          BRIDGE_MULT=5, CONTEXT_SECONDS=5, TOLERANCE=5):
    """
    Handles N-stage sequential handovers (0->1, 1->2, ... N-1->N)
    to identify Frames of Interest (FOI) across a complex LTL chain.
    """
    fps = video_info["fps"]
    frame_count = video_info["frame_count"]
    
    # Configuration
    MAX_GAPS = BRIDGE_MULT * fps
    WINDOW = CONTEXT_SECONDS * fps
    tolerance = TOLERANCE * fps

    # Start with the ground truth from the automaton
    final_set = set(automaton_foi)
    valid_segments = set()

    # Iterate through each temporal transition (Stage i -> Stage i+1)
    # If all_detections has 3 sets, we check (0,1) and (1,2)
    if len(all_detections) == 1:
        stage_1 = sorted(list(all_detections[0]))
        segments = group_with_gaps(stage_1, max_gaps=MAX_GAPS)
        for cluster in segments:
            t_start = min(cluster)
            t_end = max(cluster)
            
            event_before = {idx for idx in stage_1 if (t_start - max(WINDOW, int(target_window_before / 2) * fps)) <= idx <= t_end}
            event_after = {idx for idx in stage_1 if t_start <= idx <= (t_end + max(WINDOW, int(target_window_after / 2) * fps))}
            
            valid_segments.update(event_before)
            valid_segments.update(event_after)
    else:
        for i in range(len(all_detections) - 1):
            p_indices = all_detections[i]
            q_indices = all_detections[i+1]

            if not p_indices or not q_indices:
                continue

            # Find where Stage i hands over to Stage i+1
            handover_all = find_soft_handover(p_indices, q_indices, tolerance)

            if handover_all:
                handover_sorted = sorted(list(handover_all))
                # Group handover points into clusters to find distinct event boundaries
                handover_clusters = group_with_gaps(handover_sorted, max_gaps=MAX_GAPS)
                
                for cluster in handover_clusters:
                    t_start = min(cluster)
                    t_end = max(cluster)
                    
                    # Capture context around the transition
                    # event_p: Context before and during the handover
                    event_p = {idx for idx in p_indices if (t_start - max(WINDOW, target_window_before)) <= idx <= t_end}
                    # event_q: Context during and after the handover
                    event_q = {idx for idx in q_indices if t_start <= idx <= (t_end + max(WINDOW, target_window_after))}
                    
                    valid_segments.update(event_p)
                    valid_segments.update(event_q)

    # Union the segments identified by handovers with the automaton's path
    final_set.update(valid_segments)

    #     # Intersection preserves formal guarantees from the automaton
    # # Densify so sparse samples intersect at the range level
    # if final_set and valid_segments:
    #     dense_automaton = densify(final_set, max_gaps=MAX_GAPS)
    #     dense_segments = densify(valid_segments, max_gaps=MAX_GAPS)
    #     final_set = dense_segments & dense_automaton
    #     if not final_set:
    #         final_set = dense_automaton
    # elif not final_set:
    #     final_set = valid_segments

    # # Fallback: If still empty, use proposition detections only

    # Fallback: If no sequential handovers were found, take the union of all hits
    if not final_set:
        final_set = set().union(*all_detections)
    
    if not final_set:
        return [-1]

    # Group the final indices to create continuous video ranges
    final_foi_groups = group_with_gaps(sorted(list(final_set)), max_gaps=MAX_GAPS)
    
    # Convert clusters to (start, end) tuples for the cropper
    return [(min(g), max(g)) for g in final_foi_groups]

# ---------------------------------------------------------------- segment output strategies

SEGMENT_MODES = ("single", "intersect", "union", "auto")


def resolve_segment_mode(mode, num_stages):
    """``auto`` picks ``intersect`` for Until specs (one P->Q moment) and ``union``
    for Until-free specs (e.g. ``F a & F b`` sequence / ``F a | F b`` selection
    questions, whose evidence sits at several separate moments)."""
    if mode not in SEGMENT_MODES:
        raise ValueError(f"Unknown segment_mode {mode!r}, expected one of {SEGMENT_MODES}")
    if mode == "auto":
        return "intersect" if num_stages >= 2 else "union"
    return mode


def single_segment(all_detections, automaton_foi, fps, context_seconds):
    """Upstream NeuS-QA output: one span where the automaton and the detections agree."""
    if not automaton_foi or not any(all_detections):
        return [-1]
    detections_foi = intersection_with_gaps(all_detections, max_gaps=fps * context_seconds)
    detections_range = set(range(int(min(detections_foi)), int(max(detections_foi)) + 1))
    foi = set(automaton_foi) & detections_range
    if not foi:
        return [-1]
    return [[min(foi), max(foi)]]


def fallback_segments(all_detections, original_indices, fps, max_gap_seconds=15):
    """Segments from raw detections, or from every sampled frame when nothing was detected."""
    source_indices = sorted(set().union(*all_detections))
    if not source_indices:
        source_indices = sorted(int(x) for x in original_indices)
    if not source_indices:
        return [-1]
    groups = group_with_gaps(source_indices, max_gaps=max_gap_seconds * fps)
    return [(min(g), max(g)) for g in groups]


def pad_and_merge(foi, target_window, fps, frame_count, bridge_seconds):
    """Pad every segment by the target window (seconds) and merge segments closer than ``bridge_seconds``."""
    if not foi or foi == [-1]:
        return [-1]
    tw_before, tw_after = resolve_target_window(target_window)
    final_ranges = sorted(
        (max(0, int(min(group) + tw_before * fps)), min(frame_count - 1, int(max(group) + tw_after * fps)))
        for group in foi
    )
    max_gaps = bridge_seconds * fps
    merged = []
    curr_start, curr_end = final_ranges[0]
    for next_start, next_end in final_ranges[1:]:
        if next_start <= curr_end + max_gaps:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged


def compute_segments(mode, video_info, all_detections, automaton_foi, original_indices, target_window,
                     bridge_seconds, context_seconds, tolerance_seconds):
    """Turn per-stage detections and automaton frames into frames of interest.

    Returns (foi, merged): the raw segments and the padded, merged ranges sent to VQA.
    ``mode`` must already be resolved (not ``auto``).
    """
    fps = video_info["fps"]
    frame_count = video_info["frame_count"]

    if mode == "single":
        foi = single_segment(all_detections, automaton_foi, fps, context_seconds)
        return foi, pad_and_merge(foi, target_window, fps, frame_count, bridge_seconds)

    if mode == "intersect":
        later_stages = set().union(*all_detections[1:]) if len(all_detections) > 1 else set()
        foi = reconcile_sparse_ltl(
            video_info=video_info,
            p_indices=all_detections[0],
            q_indices=later_stages,
            automaton_foi=automaton_foi,
            target_window=target_window,
            BRIDGE_MULT=bridge_seconds,
            CONTEXT_SECONDS=context_seconds,
            TOLERANCE=tolerance_seconds,
        )
    elif mode == "union":
        tw_before, tw_after = resolve_target_window(target_window)
        foi = reconcile_dynamic_ltl(
            video_info=video_info,
            all_detections=all_detections,
            automaton_foi=automaton_foi,
            target_window_before=tw_before,
            target_window_after=tw_after,
            BRIDGE_MULT=bridge_seconds,
            CONTEXT_SECONDS=context_seconds,
            TOLERANCE=tolerance_seconds,
        )
    else:
        raise ValueError(f"Unresolved segment mode {mode!r}")

    if not foi or foi == [-1]:
        foi = fallback_segments(all_detections, original_indices, fps)
    return foi, pad_and_merge(foi, target_window, fps, frame_count, bridge_seconds)
