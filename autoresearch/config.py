"""The fixed rules of the autoresearch loop (frozen 16 Nov 2026; decisions in goals/communication.md, E36 / E37).

The agents may NEVER edit this file, score.py, judge.py, loop.py or ailcup_client.py: they are the judge.
Change a rule only by hand, between loop runs, and write the change into docs/research_log.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BC_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LoopConfig:
    # ---------- the judge: the real AIL Cup (autoresearch/ailcup_client.py)
    tracks: tuple[str, ...] = ("fiit-oval", "hairpin-valley", "danube-chicane")
    laps: int = 2
    judge_wait_max_s: float = 15 * 60          # one test drive; a longer wait = the platform failed, retry
    platform_retries: int = 2                  # retries of a run with sim_error (the platform, not the car)

    # ---------- the score: sum of the total times of all tracks, lower is better (score.py)
    dnf_penalty_s: float = 300.0               # a DNF track counts as 300 s: > the slowest real 2 laps (125 s), so a crash never wins
    baseline_score_s: float = 337.81           # E37: mean of 3 runs per track, wm2_c4 + simple_controller v5
    keep_margin_s: float = 1.5                 # E37: the baseline spread was 1.16 s; keep only if best - score > this

    # ---------- what the agent may change (phase 1: the controller only, D16 / communication.md 16 Nov)
    editable_files: tuple[str, ...] = ("ail_cup/simple_controller.py",)
    model_path: str = "ail_cup/wm2_c4_value_garage.npz"   # fixed in phase 1
    max_code_bytes: int = 64 * 1024            # the Garage limit

    # ---------- bookkeeping
    branch_prefix: str = "autoresearch/run"    # kept versions: local commits on autoresearch/run<N>, no push
    base_branch: str = "autoresearch/base"     # EVERY run starts here (baseline v5 + this judge); main is not touched
    runs_dir: str = "autoresearch/runs"        # runs/run<N>/results.tsv + exp_<nr>.json
    max_experiments: int = 20                  # the same budget for every agent = a fair comparison

    def path(self, relative: str) -> Path:
        return BC_ROOT / relative


CONFIG = LoopConfig()
