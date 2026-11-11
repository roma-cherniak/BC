# Monthly goal: November 2026

## Main goal
Build an **autoresearch loop for self-driving models on the JEPA architecture**: AI agents (Claude, Gemini, ...)
improve the driving model on their own, and the agents are compared with each other.

The result at the end of the work is **a graph of the metric over the experiments**, like the progress graph in
Karpathy's `autoresearch` on GitHub: experiment number on the x axis, the metric on the y axis, kept improvements
marked, a "best so far" line, and one curve per agent.

## The two steps
| Step | Week | Result |
|---|---|---|
| 1. A solid JEPA baseline car | [Week 1](week.md#week-1-step-1-a-solid-jepa-baseline-car): Tue 10 → Tue 17 Nov | crash-free laps on `practice` + `compete` and in the AIL Cup; a fixed `eval` + baseline score |
| 2. The autoresearch loop | [Week 2](week.md#week-2-step-2-the-autoresearch-loop): Tue 17 → Tue 24 Nov | the loop makes the car **faster**; several agents compared on one graph |

Tue 25 – Sun 30 Nov: buffer for anything that slips.

Speed is found by the loop, not by hand: step 1 only has to be reliable.

## The metric (the y axis of the final graph)
Decided (2026-11-11): the **sim drive** decides inside the loop (automatic). The **AIL Cup** decides for each agent's
best model (manual upload). Together they rank the agents.

<!-- TODO(human): define the formula of the SIM score the loop optimizes.
     Write: what is measured, on which tracks, how crashes count, lower or higher is better,
     and how long one measurement may take. -->

## How I will know it worked
| Check | Now | Target |
|---|---|---|
| Tracks driven without a crash | practice track only | every track |
| Lap time / speed | throttle 0.12 | as fast as the car can go |
| AIL Cup Garage | wm2_self1 tried 10 Nov (controllers v11–v16): no finished lap (hit the wall, or over the 250 ms cap) | full laps |
| Autoresearch loop | does not exist | runs alone for N experiments |
| Agents compared | 0 | 2 or more (Claude, Gemini, ...) |
| Final graph | none | metric vs experiment, one curve per agent |

## End-of-month review
- What worked:
- What did not:
- Next month:
