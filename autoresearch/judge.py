"""The judge: one controller + model -> runs on every track of the real AIL Cup -> one score.

Steps (rules from config.py):
  1. read the files on this PC and run the Garage's code check; a failed check = every track a DNF (the agent's fault)
  2. upload the controller + model ONCE (one saved version)
  3. one test drive per track, ONE AFTER ANOTHER (3 at once scored 2.5 s worse, E38: the controller's time budget)
  4. a run the PLATFORM broke (sim_error, network) is started again, up to CONFIG.platform_retries times
  5. score.py turns the rows into one number

Usage (login in AILCUP_LOGIN / AILCUP_PASSWORD, like ailcup_client):
    venv\\Scripts\\python -m autoresearch.judge [controller.py] [--label text] [--out result.json]
The agents may NEVER edit this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from autoresearch.ailcup_client import AilCupClient, AilCupError, read_inputs, to_row
from autoresearch.config import CONFIG, LoopConfig
from autoresearch.score import NotScorable, Score, score


@dataclass(frozen=True)
class Verdict:
    score: Score
    rows: dict[str, dict]            # one ailcup_client row per track
    code_ok: bool
    code_problems: list


def _try_start(client: AilCupClient, code: str, track: str, submission_id: int | None,
               config: LoopConfig) -> int | None:
    """Start a test drive now; None if the site refused (then _drive_one starts it again, with retries)."""
    try:
        return client.start_test_drive(code, config.laps, track, submission_id)
    except AilCupError as e:
        print(f"judge: {track}: start failed, will retry: {e}", file=sys.stderr)
        return None


def _drive_one(client: AilCupClient, code: str, track: str, submission_id: int | None,
               label: str, check: dict, config: LoopConfig, job_id: int | None = None) -> dict:
    """One track until a scorable row: the car's own result, or an error after the retries."""
    for attempt in range(config.platform_retries + 1):
        try:
            if job_id is None:
                job_id = client.start_test_drive(code, config.laps, track, submission_id)
            row = to_row(client.wait_run(job_id, config.judge_wait_max_s), f"{label} {track}", check)
        except AilCupError as e:
            row = {"run_id": job_id, "sim_error": f"client: {e}"}
        if not row.get("sim_error"):
            return row
        print(f"judge: {track} attempt {attempt + 1}: platform error: {row['sim_error']}", file=sys.stderr)
        job_id = None
    raise NotScorable(f"{track}: the platform failed {config.platform_retries + 1} times")


def judge(controller: Path, model: Path | None, label: str, client: AilCupClient,
          config: LoopConfig = CONFIG) -> Verdict:
    code, model_bytes = read_inputs(controller, model)
    check = client.check_code(code)
    if not check.get("ok", False):                      # the agent broke the code: every track is a DNF
        rows = {t: {"finished": False, "dnf_reason": "code_check", "code_check": check} for t in config.tracks}
        return Verdict(score(rows, config), rows, False, check.get("problems", []))

    submission_id = client.upload(code, model_bytes, model.stem if model else "", label)
    rows = {}
    for t in config.tracks:              # ONE AFTER ANOTHER, like the baseline (E37): parallel runs were slower (E38)
        job_id = _try_start(client, code, t, submission_id, config)
        rows[t] = _drive_one(client, code, t, submission_id, label, check, config, job_id)
    return Verdict(score(rows, config), rows, True, [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a controller on every track of the real AIL Cup")
    ap.add_argument("controller", nargs="?", type=Path, default=CONFIG.path(CONFIG.editable_files[0]))
    ap.add_argument("--model", type=Path, default=CONFIG.path(CONFIG.model_path))
    ap.add_argument("--label", default="judge")
    ap.add_argument("--out", type=Path, help="save the whole verdict as JSON")
    args = ap.parse_args(argv)
    client = AilCupClient()
    try:
        read_inputs(args.controller, args.model)       # wrong paths fail before the login
        client.login()
        verdict = judge(args.controller, args.model, args.label, client)
    except (AilCupError, NotScorable) as e:
        print(f"judge: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("judge: stopped by the user (started runs may still be running on the site)", file=sys.stderr)
        return 130
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(asdict(verdict), indent=1), encoding="utf-8")
    s = verdict.score
    print(json.dumps({"score_s": s.total_s, "per_track_s": s.per_track_s, "dnf": list(s.dnf_tracks),
                      "code_ok": verdict.code_ok}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
