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


def parse_sample_schedule(spec: str) -> list[tuple[float, float]]:
    """Parse ``"1.0:60,0.5:160,0.2"`` into ``[(1.0, 60), (0.5, 160), (0.2, inf)]``.

    Each ``rate:until`` stage samples at ``rate`` fps until ``until`` seconds; the last
    stage has no bound and applies for the rest of the video.
    """
    stages = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("sample_schedule is empty")
    for i, part in enumerate(parts):
        rate_s, _, until_s = part.partition(":")
        rate = float(rate_s)
        until = float(until_s) if until_s else math.inf
        if rate <= 0 or not math.isfinite(rate):
            raise ValueError(f"sample_schedule rate must be positive: {part!r}")
        if (until_s == "") != (i == len(parts) - 1):
            raise ValueError(f"only the last sample_schedule stage may omit its end time: {spec!r}")
        if stages and until <= stages[-1][1]:
            raise ValueError(f"sample_schedule end times must increase: {spec!r}")
        stages.append((rate, until))
    return stages


def schedule_frame_times(stages: list[tuple[float, float]], duration: float) -> list[float]:
    """Frame times in ``[0, duration)`` sampled at each stage's rate."""
    times, t = [], 0.0
    for rate, until in stages:
        while t < min(until, duration) - 1e-9:
            times.append(t)
            t += 1.0 / rate
    return times


class GridPlan:
    """Streaming counterpart of uniform sampling: dense early, thinned as the video grows.

    Frames come in Qwen temporal pairs ``(t, t + pair_gap)`` whose start ``t`` lies on a grid
    of spacing ``spacing(T) = g0 * 2^j`` (capped at ``smax``), the smallest such value that
    keeps about ``n_groups`` groups over ``[0, T]``. The spacing only grows, so grids are
    nested: a group dropped at time ``T`` is never needed again, and a streamed frame is only
    decoded if its group is on the grid when it arrives.
    """

    def __init__(self, n_groups: int, smax: float, g0: float = 2.0, pair_gap: float = 1.0,
                 spread: bool = False):
        if n_groups <= 0 or smax < g0:
            raise ValueError("grid plan needs n_groups > 0 and smax >= g0")
        self.n_groups, self.smax, self.g0, self.pair_gap = n_groups, smax, g0, pair_gap
        # spread: a pair's second frame sits half the spacing (at encode time) after its first, so a
        # temporal group covers its whole grid cell instead of pair_gap seconds followed by a gap.
        self.spread = spread

    @classmethod
    def parse(cls, spec: str) -> "GridPlan":
        """``"grid:N:smax"`` plus an optional pair mode: ``spread`` or ``dup``.

        ``dup`` feeds each grid frame twice (pair gap 0), so a Qwen temporal group holds one
        moment instead of blending two, e.g. ``"grid:32:8:dup"``.
        """
        parts = spec.split(":")
        if len(parts) not in (3, 4) or (len(parts) == 4 and parts[3] not in ("spread", "dup")):
            raise ValueError(f"bad grid sample_schedule: {spec!r}")
        mode = parts[3] if len(parts) == 4 else None
        return cls(int(parts[1]), float(parts[2]), spread=mode == "spread",
                   pair_gap=0.0 if mode == "dup" else 1.0)

    def spacing(self, t: float) -> float:
        s = self.g0
        while s * 2 <= self.smax and t > self.n_groups * s:
            s *= 2
        return s

    def on_grid(self, anchor: float, t: float) -> bool:
        step = round(self.spacing(t) / self.g0)
        return round(anchor / self.g0) % step == 0

    def frame_times(self, duration: float) -> tuple[list[float], list[float]]:
        """(frame times, group anchor of each frame) for frames decoded over ``[0, duration)``."""
        times, anchors = [], []
        k = 0
        while k * self.g0 < duration:
            t = k * self.g0
            if self.on_grid(t, t):
                gap = self.spacing(t) / 2 if self.spread else self.pair_gap
                for ft in (t, t + gap):
                    if ft < duration:
                        times.append(ft)
                        anchors.append(t)
            k += 1
        return times, anchors


class DupSchedule:
    """``dup:<rate stages>``: stream each scheduled frame twice (see BaseVQA.load_scheduled_video)."""

    def __init__(self, stages: list[tuple[float, float]]):
        self.stages = stages
