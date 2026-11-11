# Weekly goals: November 2026

Part of [month.md](month.md): the autoresearch loop for self-driving models on JEPA.
Weeks run **Tuesday → Tuesday**. Tue 25 – Sun 30 Nov is a **buffer** for anything that slips.

The detailed plan of each day is in [days/](days/) (one file per day, with a Log for the handover). New code goes into [autoresearch/](../autoresearch/README.md).

Every day: **Claude** = what I do, **You** = what you do or decide, **Night** = what the PC runs alone.
The sim and a training never run at the same time (4 GB GPU).

---

## Week 1 (step 1): a solid JEPA baseline car
**Tue 10 → Tue 17 Nov**

### Goal
A wm2 car that finishes laps **without crashes** on `practice` **and** `compete` in the sim, and in the AIL Cup,
at a moderate speed. Making it the **fastest** is the job of the loop in week 2.
At the end there is also a **fixed evaluation script** and a **baseline score**, which is the first point on the final graph.

### Days
| Day | Claude | You | Night |
|---|---|---|---|
| Tue 10 ✓ | | | self2 loop: collect 2.1 h, train, heads (done) |
| **Wed 11** | compare self2 vs self1 from the existing drive logs; check that `compete` opens and that the drivers accept it; commit | answer the open questions in [communication.md](communication.md); tell me what AIL Cup tracks and rules exist | collect random + self-driving trips on `compete` |
| Thu 12 | drive self2 on `compete` without retraining (does it handle a track it has never seen?); export self2 for the AIL Cup | upload to the Garage, report laps / crashes | night loop on both tracks: collect → train → heads |
| Fri 13 | `plan_test` + sim drives of the new model on both tracks; find where it crashes | look at the crash places with me, decide what to fix first | fix-driven collection (more trips where it failed) |
| Sat 14 | speed ladder 0.12 → 0.15 → 0.18 on both tracks; find the limit | choose the baseline speed | train the fix |
| Sun 15 | fix the biggest crash cause; AIL Cup export #2 | Garage test #2 | self-driving round + train |
| Mon 16 | write `eval` (fixed tracks, fixed time, one score); run the baseline **3×** to measure the noise | approve the score formula ([month.md](month.md) metric) | spare: rerun what failed |
| **Tue 17** | week 1 report; the baseline score and its noise | **review: is the baseline good enough to start the loop?** | |

### Done when
- [ ] 0 crashes for 5 minutes on `practice` and `compete`
- [ ] Full laps in the AIL Cup
- [ ] `eval` exists and gives one number; the baseline was measured 3× and its noise is known

### Questions for this week
- What tracks does the AIL Cup have, and what are its limits (time per call, file size, class rules)?
- Did wm2 ever drive `compete`, or only `practice`?
- If the car is not crash-free by Sun 15: start week 2 on a weaker baseline, or move week 2 by 2 days?

---

## Week 2 (step 2): the autoresearch loop
**Tue 17 → Tue 24 Nov**

### Goal
An AI agent changes one editable part, runs **one experiment in a fixed time** (short training + `eval` in the sim),
keeps the change only if the score improves beyond the noise, and repeats. The same loop runs with different agents
from the same baseline, and each agent's **best** model also gets an AIL Cup result.

### Days
| Day | Claude | You | Night |
|---|---|---|---|
| **Tue 17** | study Karpathy's `autoresearch`; propose the design: frozen files (eval, data split, tracks), the file agents may edit, time budget, size limit | decide the frozen / editable split and the budget | baseline repeats if noise is still unclear |
| Wed 18 | build the loop runner: train → eval → `results.tsv` → keep / revert with git; test it with a dummy agent | review the runner | dummy run all night (does it survive 8 h alone?) |
| Thu 19 | write `program.md` (the same instructions for every agent); first Claude run, supervised | watch the first experiments, fix the rules | **Claude** run 1 |
| Fri 20 | install + set up Gemini CLI on the same `program.md`; analyse Claude's night | provide the Gemini access / account | **Gemini** run 1 |
| Sat 21 | the third agent (if any); export each agent's best for the AIL Cup | Garage tests of the bests | **Claude** run 2 |
| Sun 22 | plotting script: score vs experiment, kept points, best-so-far line, one curve per agent | | **Gemini** run 2 |
| Mon 23 | the final graph; rerun the top results to check they are not noise | AIL Cup tests of the final bests | spare / third agent |
| **Tue 24** | week 2 report + the final graph | **review** | |

### Done when
- [ ] The loop ran a whole night alone
- [ ] At least 2 agents finished the **same** budget from the **same** baseline
- [ ] The final graph (sim score) + an AIL Cup result for each agent's best

### Questions for this week
- Which agents exactly (Claude Code, Gemini CLI, Codex, ...), and do you have access to each one?
- What may an agent change: only the training settings, the model code, the planner, or all of them?
- How many experiments per agent (equal budget = fair comparison)?

---

## Weekly reviews
- Week 1 (Tue 17):
- Week 2 (Tue 24):
