"""The autoresearch loop (v1, 16 Nov 2026; the user's design, goals/communication.md):

    repeat N times:
      1. the agent edits ail_cup/simple_controller.py   (it reads program.md + results.tsv)
      2. the judge runs it on the real AIL Cup            (judge.py: 3 tracks x 2 laps -> one score)
      3. one line in results.tsv
      4. better than the best by > the noise?  yes: git commit on autoresearch/run<N>   no: undo the change

Safety: the agent may change ONLY CONFIG.editable_files. Any other changed file stops the loop (nothing is undone
outside the editable files, so you can look at what happened). The loop never pushes, never touches main, and never
switches to an existing branch: a new run is created from CONFIG.base_branch (autoresearch/base), so every agent
starts from the same baseline.

Usage (from BC, login in AILCUP_LOGIN / AILCUP_PASSWORD):
    venv\\Scripts\\python -m autoresearch.loop --run 1 --agent claude --experiments 20
    venv\\Scripts\\python -m autoresearch.loop --run 0 --agent dummy --experiments 2   (tests the plumbing)
A run that stopped continues where it ended when started again with the same --run.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from autoresearch.config import BC_ROOT, CONFIG, LoopConfig
from autoresearch.score import NotScorable, is_better

AGENT_TIMEOUT_S = 20 * 60
COLUMNS = ["exp", "agent", "score_s", *CONFIG.tracks, "dnf", "kept", "best_s", "change", "commit"]

Agent = Callable[[str, Path], str]            # (prompt, BC folder) -> the agent's answer text
JudgeFn = Callable[[Path, str], "object"]     # (controller, label) -> judge.Verdict


class LoopStopped(RuntimeError):
    """A rule was broken (a forbidden file changed, git failed): the loop stops so the user can look."""


# ---------------------------------------------------------------- agents
def claude_agent(prompt: str, root: Path) -> str:
    """Claude Code in print mode, allowed to read and edit files only (no shell)."""
    exe = shutil.which("claude")
    if exe is None:
        raise LoopStopped("the `claude` command is not installed")
    done = subprocess.run([exe, "-p", prompt, "--allowedTools", "Read,Edit,Grep,Glob",
                           "--permission-mode", "acceptEdits"], cwd=root, capture_output=True,
                          text=True, encoding="utf-8", timeout=AGENT_TIMEOUT_S)
    if done.returncode != 0:
        raise LoopStopped(f"claude failed ({done.returncode}): {done.stderr[-500:]}")
    return done.stdout


DUMMY_KNOBS = ("SIDE_MARGIN", "START_SPEED", "BUDGET_SHARE")


def dummy_agent(prompt: str, root: Path, rng: random.Random = random.Random()) -> str:
    """Tests the plumbing: scales one number of the controller by 0.9 - 1.1."""
    path = root / CONFIG.editable_files[0]
    text = path.read_text(encoding="utf-8")
    knob = rng.choice(DUMMY_KNOBS)
    m = re.search(rf"^{knob} = ([0-9.]+)", text, flags=re.M)
    if m is None:
        return "CHANGE: nothing (knob not found)"
    new = round(float(m.group(1)) * rng.uniform(0.9, 1.1), 3)
    path.write_text(text[:m.start(1)] + str(new) + text[m.end(1):], encoding="utf-8")
    return f"CHANGE: {knob} {m.group(1)} -> {new} (dummy)"


AGENTS: dict[str, Agent] = {"claude": claude_agent, "dummy": dummy_agent}


def change_line(answer: str) -> str:
    lines = [l for l in answer.splitlines() if l.strip().startswith("CHANGE:")]
    return (lines[-1].split("CHANGE:", 1)[1].strip() if lines else "(no CHANGE line)")[:300]


# ---------------------------------------------------------------- git
def git(root: Path, *args: str, strip: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        raise LoopStopped(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip() if strip else done.stdout


def changed_paths(root: Path) -> set[str]:
    """Every changed or new file. Lines are 'XY path' (or 'XY old -> new'): the first 3 characters are the status."""
    out = git(root, "status", "--porcelain", "-uall", strip=False)   # strip would eat the space of ' M path'
    return {line[3:].split(" -> ")[-1].strip('"') for line in out.splitlines() if len(line) > 3}


def start_branch(root: Path, branch: str, base: str) -> None:
    """A new run starts from `base` (the same point for every agent); an existing run continues on its branch."""
    current = git(root, "branch", "--show-current")
    if current == branch:
        return
    if git(root, "branch", "--list", branch):
        raise LoopStopped(f"branch {branch} exists but is not checked out: switch to it yourself first")
    if not git(root, "branch", "--list", base):
        raise LoopStopped(f"the base branch {base} does not exist: create it at the baseline commit first")
    git(root, "switch", "-c", branch, base)   # git refuses by itself if this would overwrite an uncommitted change


# ---------------------------------------------------------------- results.tsv
@dataclass
class RunState:
    next_exp: int
    best_s: float


def load_or_start(results: Path, config: LoopConfig) -> RunState:
    if not results.exists():
        results.parent.mkdir(parents=True, exist_ok=True)
        write_row(results, {"exp": 0, "agent": "baseline", "score_s": config.baseline_score_s, "kept": "yes",
                            "best_s": config.baseline_score_s, "change": "baseline (E37)"})
        return RunState(1, config.baseline_score_s)
    with results.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return RunState(int(rows[-1]["exp"]) + 1, float(rows[-1]["best_s"]))


def write_row(results: Path, row: dict) -> None:
    is_new = not results.exists()
    with results.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


# ---------------------------------------------------------------- the loop
def make_prompt(program: str, results: Path, root: Path, exp: int, last: int, best_s: float) -> str:
    return (f"{program}\n\n## This experiment\nExperiment {exp} of {last}. Best score so far: {best_s:.2f} s.\n"
            f"The results so far: `{results.relative_to(root).as_posix()}`.\n")


def one_experiment(exp: int, agent_name: str, agent: Agent, judge_fn: JudgeFn, state: RunState,
                   root: Path, results: Path, prompt: str, config: LoopConfig) -> dict:
    editable = set(config.editable_files)
    before = changed_paths(root)
    answer = agent(prompt, root)
    touched = changed_paths(root) - before
    forbidden = touched - editable
    if forbidden:
        raise LoopStopped(f"experiment {exp}: the agent changed forbidden files: {sorted(forbidden)}")
    row = {"exp": exp, "agent": agent_name, "change": change_line(answer), "kept": "no", "best_s": state.best_s}
    if not touched:
        return {**row, "score_s": "", "dnf": "no change"}
    try:
        verdict = judge_fn(root / config.editable_files[0], f"{config.branch_prefix} exp {exp}")
    except NotScorable as e:
        git(root, "checkout", "--", *editable)
        return {**row, "score_s": "", "dnf": f"not scored: {e}"[:200]}
    s = verdict.score
    row.update({"score_s": s.total_s, **s.per_track_s, "dnf": ",".join(s.dnf_tracks)})
    if is_better(s.total_s, state.best_s, config):
        git(root, "add", "--", *editable)
        git(root, "commit", "-m", f"autoresearch exp {exp}: {s.total_s:.2f} s ({agent_name}): {row['change']}")
        state.best_s = s.total_s
        return {**row, "kept": "yes", "best_s": s.total_s, "commit": git(root, "rev-parse", "--short", "HEAD")}
    git(root, "checkout", "--", *editable)
    return row


def run_loop(run: int, agent_name: str, experiments: int, judge_fn: JudgeFn, root: Path = BC_ROOT,
             config: LoopConfig = CONFIG, agents: dict[str, Agent] = AGENTS) -> Path:
    agent = agents[agent_name]
    if changed_paths(root) & set(config.editable_files):
        raise LoopStopped(f"{config.editable_files} has uncommitted changes: commit or undo them first")
    start_branch(root, f"{config.branch_prefix}{run}", config.base_branch)
    results = root / config.runs_dir / f"run{run}" / "results.tsv"
    state = load_or_start(results, config)
    program = (root / "autoresearch" / "program.md").read_text(encoding="utf-8")
    last = state.next_exp + experiments - 1
    for exp in range(state.next_exp, last + 1):
        prompt = make_prompt(program, results, root, exp, last, state.best_s)
        row = one_experiment(exp, agent_name, agent, judge_fn, state, root, results, prompt, config)
        write_row(results, row)
        print(f"exp {exp}: score {row.get('score_s') or '-'}  kept {row['kept']}  best {state.best_s:.2f}  "
              f"{row['dnf'] or ''}  | {row['change']}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The autoresearch loop on the real AIL Cup")
    ap.add_argument("--run", type=int, required=True, help="run number: branch autoresearch/run<N>")
    ap.add_argument("--agent", choices=sorted(AGENTS), required=True)
    ap.add_argument("--experiments", type=int, default=CONFIG.max_experiments)
    args = ap.parse_args(argv)

    from autoresearch.ailcup_client import AilCupClient, AilCupError
    from autoresearch.judge import judge

    client = AilCupClient()
    model = CONFIG.path(CONFIG.model_path)
    try:
        client.login()
        results = run_loop(args.run, args.agent, args.experiments,
                           lambda controller, label: judge(controller, model, label, client))
    except (LoopStopped, AilCupError) as e:
        print(f"loop stopped: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if changed_paths(BC_ROOT) & set(CONFIG.editable_files):       # the unfinished experiment is undone
            git(BC_ROOT, "checkout", "--", *CONFIG.editable_files)
        print("loop stopped by the user; start it again with the same --run to continue", file=sys.stderr)
        return 130
    print(f"done: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
