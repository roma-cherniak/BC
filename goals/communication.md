# Communication: me ↔ Claude

How it works:
- **You** write in "To Claude", with the newest message at the top and a date. Write anything: tasks, ideas, corrections, "never do X".
- **Claude** reads this file at the start of every session, answers in "From Claude", and moves anything that is finished to "Archive".
- Questions that need your decision go in "Open questions". Write your answer under the question.

---

## To Claude
<!-- example:
### 2026-11-12
Today only offline checks, the sim is busy.
-->


## From Claude
### 2026-11-11
- I created [month.md](month.md), [week.md](week.md) and [day.md](day.md). The goals in them are **drafts** that I wrote from the project state. Please change them so they are yours.
- `autoresearch/eval/drive_stats.py` is written except for one function that is yours: `wall_clearance` (look for `TODO(human)`).
- I found `research/` (`run_trial.py`, `loop1.py`): a driver-tuning loop with its own judge. No plan file mentions it. I reused its rules (D15) and did not change the folder. See question 7.
- Thank you for answers 3–6. They are now in "Decisions" and in D16 of the research log; the questions are in "Archive". Three things follow from them:
  1. **"An agent may change everything"**: today `wm2/drive.py` counts its own crashes and laps and writes the log the judge reads. An agent that may edit the driver could then edit its own score. So the frozen judge must own the sim connection and the log, and the candidate only returns an action. I will plan this for Mon 16 (`drive_eval.py`).
  2. **Claude Pro + free APIs**: both have usage limits that can stop a night run. A fair budget is the same **number of experiments** per agent, not the same hours. The dummy night of Wed 18 will show how far one night gets.
  3. **The dates do not move**: then the scope has to. My proposal: if the car is not crash-free at gas 0.12 by Sun 15, the baseline is the highest gas ceiling that **is** crash-free, even if it is slow. Say so if you disagree.
- **The night run is ready to start** (not yet tried with the compete sim itself):
  1. Open the **compete** sim: `BP\autodrive_simulator_compete_windows\autodrive_simulator\AutoDRIVE Simulator.exe`.
  2. In the `BC` folder: `venv\Scripts\python night\compete_night.py`
  3. Click "Connect" in the game within 10 minutes.
  4. Watch the first 2 minutes: `night/state_compete.txt` and `night/compete_test_a.log` (it must print speed lines).
  - What it does: 2 test drives of self2 (11 min) → random collection 3 h → self-driving 3 h → closes the sim. No training.
    If self2 crashes more than 10 times per minute in the tests, the self-driving hours go to random collection. Setup: E14.
  - Other hours: `set NIGHTC_RANDOM_HOURS=4` and `set NIGHTC_SELF_HOURS=2` before step 2. To stop early: close the sim.
- I read `BP/CHAT_CONTEXT.md` only now (it is not in the reading list of `BC/CLAUDE.md`). It already had the self1 / self2 table (now E13), and it answers question 8.

## Open questions
1. **Which project is active?** `BP/CLAUDE.md` says `autodrive-lewm/`, but the current work is in `BC/` (wm2). Should I treat `BC` as the main project?
   - Your answer: BC
2. **How should I keep the goal files?** Should I tick off finished tasks and write tomorrow's `day.md` myself, or only when you ask?
   - Your answer: 

## Decisions (stay valid until changed)
| Date | Decision |
|---|---|
| 2026-11-11 | New main goal: an autoresearch loop for self-driving JEPA models. Step 1 = the fastest JEPA car (every track, sim + AIL Cup), step 2 = the loop with different AI agents compared on one graph. See [month.md](month.md). |
| 2026-11-11 | Weeks run Tue → Tue. Week 1 = Tue 10 → Tue 17 (a solid baseline, not the fastest). Week 2 = Tue 17 → Tue 24. Buffer Tue 25 – Sun 30. |
| 2026-11-11 | Keep the wm2 architecture (no LeWM rebuild); architecture changes belong to the loop in week 2. See D13 in [docs/research_log.md](../docs/research_log.md). |
| 2026-11-11 | Metric: the sim drive inside the loop + the AIL Cup for each agent's best. You work 6+ h per day. |
| 2026-11-11 | AIL Cup: 3 tracks, unlimited uploads and tries, a model up to 50 MB and up to 64 arrays. |
| 2026-11-11 | Agents: Claude (Pro subscription) + other models through free APIs. |
| 2026-11-11 | An agent may change everything except the judge. See D16 in [docs/research_log.md](../docs/research_log.md). |
| 2026-11-11 | The dates do not move: week 2 starts Tue 17 whatever the baseline is. |
| 2026-11-11 | `research/` is archived (`archive/research/`); the loop is written new in `autoresearch/`. |
| 2026-11-11 | The 3 AIL Cup maps: an oval, a winding track, an oval with two chicanes. Pictures in `docs/ail_cup_maps/`. See D17. |
| 2026-11-11 | `explore` is not a track: data is collected only on `practice` and `compete`. |

## Archive
### Answered 2026-11-11
11. **A night on the `explore` sim build this week?**
   - Your answer (in the chat): explore is not a track, do not consider it for data collection.
10. **Are the 3 AIL Cup maps the same tracks as the sim builds** `practice` / `compete` / `explore`, or different ones?
   - Your answer (in the chat): in the sim I have different tracks.
7. **`research/` folder:** keep it as it is, move it into `autoresearch/`, or put it in `archive/`?
   - Your answer: move it to archive so you can write new better version from scratch
   - Done: it is now `archive/research/`.
9. **AIL Cup, the rest of question 3:** which are its 3 tracks?
   - Your answer: three screenshots of the maps (saved as `docs/ail_cup_maps/map1_oval.png`, `map2_winding.png`, `map3_two_chicanes.png`).
   - Time limit (found in `BP/CHAT_CONTEXT.md`): the Garage stops a controller whose control step takes more than 250 ms.
8. **The `test` drives of 10 Nov** (`drive_practice_self?_test_a/b.csv`): which `FRONT_BASE` did they use?
   - Answer (from `BP/CHAT_CONTEXT.md` and `BP/claude_tools/batch_*.txt`, not from you): the default, 0.25.
3. **AIL Cup:** which tracks, what limits (time per call, file size, class), and how many uploads per day can you do?
   - Your answer: ail cap have 3 tracks with unlimited number of apploads and tries, i can appload model up to 50 mb and up to 64 arrays , i can appload every time when its needed
4. **Agents:** which ones (Claude Code, Gemini CLI, Codex, ...), and do you have access / budget for each one?
   - Your answer: i have claude pro subscription and free api for another models
5. **What may an agent change?** Training settings only, the model code, the planner, or everything (except `eval`)?
   - Your answer: idealy it may have posibility to change everything
6. **If week 1 slips** (not crash-free by Sun 15): start week 2 on a weaker baseline, or move it by 2 days into the buffer?
   - Your answer: its not possible we must do everything to make it in date
