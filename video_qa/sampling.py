"""Frame-index selection utilities shared by video QA inference modes."""

from __future__ import annotations

import math


def frame_end_exclusive(
    *, total_frames: int, fps: float, end_time: float | None
) -> int:
    """Return the exclusive source-frame bound through ``end_time`` seconds."""
    if total_frames < 0:
        raise ValueError("total_frames must be nonnegative")
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be a positive finite number")
    if end_time is None:
        return total_frames
    if end_time < 0 or not math.isfinite(end_time):
        raise ValueError("end_time must be a nonnegative finite number")

    # Include the source frame at or immediately before the question timestamp.
    return min(total_frames, math.floor(end_time * fps) + 1)


def uniform_frame_indices(
    *,
    total_frames: int,
    num_frames: int,
    end_frame_exclusive: int | None = None,
) -> list[int]:
    """Select up to ``num_frames`` unique indices uniformly from frame zero."""
    if total_frames < 0:
        raise ValueError("total_frames must be nonnegative")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    end = total_frames if end_frame_exclusive is None else end_frame_exclusive
    end = min(total_frames, max(0, end))
    count = min(num_frames, end)
    if count == 0:
        return []
    if count == 1:
        return [0]

    # Integer arithmetic includes both endpoints and avoids duplicate indices.
    span = end - 1
    return [(index * span) // (count - 1) for index in range(count)]
