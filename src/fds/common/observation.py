"""Public RGB temporal sampling, shared by all methods."""

import numpy as np


def image_change(left, right):
    values = [
        float(
            np.mean(
                np.abs(image[::8, ::8].astype(float) - left.images[camera][::8, ::8].astype(float))
            )
        )
        for camera, image in right.images.items()
        if camera in left.images and image.shape == left.images[camera].shape
    ]
    return max(values, default=0.0)


def select_frames(frames, budget):
    """Temporal coverage + separated visual changes; never a semantic detector."""
    if len(frames) <= budget:
        return list(frames)
    if budget == 1:
        return [frames[-1]]
    selected = set(np.linspace(0, len(frames) - 1, max(2, budget // 2), dtype=int))
    separation = max(1, len(frames) // (budget * 2))
    scores = sorted(
        ((image_change(frames[i - 1], frames[i]), i) for i in range(1, len(frames))), reverse=True
    )
    for _, i in scores:
        if len(selected) >= budget:
            break
        if all(abs(i - j) >= separation for j in selected):
            selected.add(i)
        if len(selected) == budget:
            break
    # Degenerate/short streams still use their declared capacity deterministically.
    for i in np.linspace(0, len(frames) - 1, budget, dtype=int):
        if len(selected) < budget:
            selected.add(int(i))
    return [frames[i] for i in sorted(selected)]
