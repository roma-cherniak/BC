# autoresearch: the autoresearch loop for JEPA self-driving models

**All new code of the November 2026 plan goes into this folder** ([goals/month.md](../goals/month.md)).
The existing folders stay as they are: `wm2/` is the model (the thing being improved), `core/`, `data_collection/`, `checks/` are
older shared code, and `night/` holds one-off night scripts.

Run everything from the `BC` folder: `venv\Scripts\python -m autoresearch.<module>`.

## The idea (after Karpathy's `autoresearch`)
An AI agent changes the **candidate** (the only part it may edit), the **runner** trains it for a fixed time and the **judge**
drives it in the sim and gives one score. The change is kept only if the score beats the best score by more than the noise.
This repeats all night. Every agent (Claude, Gemini, ...) starts from the same baseline with the same `program.md` and the
same budget, and the final graph compares them.

## Architecture (planned: each part is built on the day shown)

```
autoresearch/
├── README.md            this file
├── __init__.py
├── config.py            FROZEN numbers: tracks, drive time per track, time budget, model-size limit   (Mon 16)
├── sim.py               start / close the simulator, choose the track (from night/ supervisor logic) (Mon 16)
├── eval/                THE JUDGE: agents never edit it
│   ├── __init__.py
│   ├── drive_stats.py   drive log CSV -> crashes/min, speed, closest wall, laps                       (Wed 11)
│   ├── drive_eval.py    drive one model for a fixed time on every track, save the logs                (Mon 16)
│   └── score.py         stats of all tracks -> ONE score (the user's formula in goals/month.md)       (Mon 16)
├── tests/               pytest tests of the judge: venv\Scripts\python -m pytest autoresearch/tests -q
├── candidate/           THE ONLY PART AGENTS MAY EDIT (contents decided Tue 17)                        (Tue 17)
├── loop/
│   ├── runner.py        one experiment: train candidate -> eval -> results.tsv -> keep / revert (git)  (Wed 18)
│   ├── program.md       the instructions every agent gets (the same for all)                           (Thu 19)
│   └── agents.md        how each agent is started (Claude Code, Gemini CLI, ...)                      (Thu 19 – Fri 20)
├── results/<agent>/results.tsv    one row per experiment: commit, score, kept, time, description
└── plot.py              the final graph: score vs experiment, kept points, best-so-far line, one curve per agent (Sun 22)
```

## Rules of the architecture
1. **The judge is separate from the candidate.** `eval/` reads only a trained run folder (`runs/<run>/`), never the candidate's
   code, so an agent cannot change how it is judged.
2. **Dependencies go one way:** `autoresearch` may import `wm2` and `core`. `wm2` and `core` never import `autoresearch`.
3. `config.py` is frozen once the baseline is measured (Mon 16). If it changes, every old score is invalid.
4. One GPU (4 GB): the runner never trains and drives the sim at the same time.
5. Every module starts with a comment block saying what it does and how to run it (the style of `wm2/`).

## Status
| Part | Status |
|---|---|
| folder + architecture | ✓ 2026-11-11 |
| `eval/drive_stats.py` + tests | in progress 2026-11-11: `wall_clearance` is not defined yet (8 of 10 tests pass) |
| everything else | see the day plans in [goals/days/](../goals/days/) |
