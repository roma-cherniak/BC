"""The score of one experiment: the results of all tracks -> one number (lower is better).

Input: one row per track from ailcup_client.to_row(). Rules from config.py:
  finished track -> its total time; DNF track -> CONFIG.dnf_penalty_s; score = the sum over CONFIG.tracks.
A row with sim_error (the platform failed, not the car) cannot be scored: the judge retries it.
The agents may NEVER edit this file.
"""
from __future__ import annotations

from dataclasses import dataclass

from autoresearch.config import CONFIG, LoopConfig


class NotScorable(ValueError):
    """The rows cannot give a fair score (a track missing, or the platform failed)."""


@dataclass(frozen=True)
class Score:
    total_s: float                    # the number the loop compares, lower is better
    per_track_s: dict[str, float]     # each track's time, or the penalty for a DNF
    dnf_tracks: tuple[str, ...]       # which tracks were a DNF (and why, in the rows)


def track_time(row: dict, config: LoopConfig = CONFIG) -> float:
    """One track's contribution: its time if it finished, the penalty if not."""
    if row.get("sim_error"):
        raise NotScorable(f"run {row.get('run_id')}: platform error: {row['sim_error']}")
    if row.get("finished") and row.get("total_time_s") is not None:
        return float(row["total_time_s"])
    return config.dnf_penalty_s


def score(rows_by_track: dict[str, dict], config: LoopConfig = CONFIG) -> Score:
    missing = [t for t in config.tracks if t not in rows_by_track]
    if missing:
        raise NotScorable(f"no result for {', '.join(missing)}")
    per_track = {t: track_time(rows_by_track[t], config) for t in config.tracks}
    dnf = tuple(t for t in config.tracks if not rows_by_track[t].get("finished"))
    return Score(total_s=round(sum(per_track.values()), 3), per_track_s=per_track, dnf_tracks=dnf)


def is_better(new_s: float, best_s: float, config: LoopConfig = CONFIG) -> bool:
    """Keep a change only if it beats the best by more than the noise (E37)."""
    return best_s - new_s > config.keep_margin_s
