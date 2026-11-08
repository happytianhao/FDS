"""Bounded temporal coverage for fresh baseline observations only."""

import math

import numpy as np

from fds.common.observation import image_change


def select_temporal_frames(frames, budget):
    """Keep endpoints and choose local visual changes near uniform time anchors.

    Each interior slot searches only one quarter of the anchor spacing on
    either side. Large motion in one part of a clip cannot consume all slots.
    No task labels, semantic detectors, stored frames or historical state.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("Frame budget must be a positive integer")
    if len(frames) <= budget:
        return list(frames)
    if budget == 1:
        return [frames[-1]]
    chosen = [0, len(frames) - 1]
    if budget == 2:
        return [frames[i] for i in chosen]
    scores = [0.0] + [image_change(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    anchors = np.linspace(0, len(frames) - 1, budget)
    radius = (len(frames) - 1) / (budget - 1) / 4
    for anchor in anchors[1:-1]:
        low = max(1, math.ceil(anchor - radius))
        high = min(len(frames) - 2, math.floor(anchor + radius))
        candidates = range(low, high + 1) if low <= high else [round(float(anchor))]
        chosen.append(max(candidates, key=lambda i: (scores[i], -abs(i - anchor), -i)))
    return [frames[i] for i in sorted(chosen)]
