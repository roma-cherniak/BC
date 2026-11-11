# Instructions for Claude Code in the BC folder

When the chat works in `BC`, this file wins over `BP/CLAUDE.md` (which is about `autodrive-lewm/`).

## Start of every session (read in this order)
1. `goals/communication.md`: the user's messages, open questions and **Decisions** (standing rules)
2. `goals/days/<today's date>.md`: today's plan. Continue the first unchecked task
3. `goals/week.md` and `goals/month.md`: where today fits (step 1 = a solid JEPA car, step 2 = the autoresearch loop)
4. `docs/research_log.md`: every decision (D) and experiment (E) so far, with evidence
5. `autoresearch/README.md`: the architecture of the new code
6. `README.md`: commands of the older parts (data, training, checks, drivers)
7. `../CHAT_CONTEXT.md` (in `BP/`, Ukrainian): the history before 11 Nov 2026 that the files above do not repeat
   (night results, AIL Cup Garage tries, how the user wants explanations). `../claude_tools/` has the older drive tools
   and the settings of each tagged test drive (`batch_*.txt`)

## During the work
- **New code goes into `autoresearch/`**, following its README. `wm2/` is the model being improved. One-off night scripts go into `night/`
- Tick the tasks `[x]` in today's day file as they are finished, and fill in its **Log** at the end of the work
  (done / not done and why / handover to tomorrow), so the next chat can continue
- If a day's task is not finished, copy it into the next day's file
- **Every decision and every experiment goes into `docs/research_log.md`** (D<n> / E<n>, with date, reason and evidence
  from a file). It is the starting point of the project documentation: never delete entries, mark them "replaced by"
- Questions for the user go into `goals/communication.md` → "Open questions", and are also asked in the chat
- Be critical: if a plan is not realistic, say so and propose a fix

## Rules
- Never run the simulator and a training at the same time (4 GB GPU)
- Ask before deleting files, committing, or changing system settings
- Run everything from the `BC` folder with `venv\Scripts\python -m ...`
