# AIL Cup web API (found 16 Nov 2026)

Site: https://ailcup.ail-lab.fiit.stuba.sk. Not documented; this is read from the site's own JavaScript
(`/static/js/app.js`, `ide.js`, `versions.js`). The backend is FastAPI (`{"detail": ...}` errors).

## Auth
- `POST /api/auth/login` sets a session cookie; `GET /api/auth/me` = who is logged in.
- Every request needs the header `x-ailcup-csrf: 1` (without it: `{"detail":"CSRF check failed"}`) + the cookie.
- Never put the password or cookie in a file in git: read them from environment variables.

## Garage (pro class) calls
| Call | Body | Returns |
|---|---|---|
| `POST /api/code/check` | `{code}` | upload check (allowed imports, errors) |
| `POST /api/submissions` | `{label, code, folder_id}` | a saved version `s` (`s.id`) |
| `PUT/POST /api/model-upload/{s.id}/{name}` | the `.npz` bytes | attaches the model to the version |
| `POST /api/test-drive` | `{code, laps, track?, model_from: s.id}` | `{job_id}` |
| `GET /api/test-drive` (`?before=<id>` = older page) | | the runs + quota (`per_day`, `left`; no `per_day` = unlimited) |
| `GET /api/test-drive/{id}/track`, `/api/jobs/{id}/replay`, `/api/jobs/{id}/export.npz` | | track, replay, run data |

## Public
- `GET /api/public/rules`: control 20 Hz, contact penalty 20 s, DNF reasons (crash, compute_timeout, timeout, ...).

## Open before use in the loop
- Do the competition rules allow scripted test drives? (ask the organisers)
- The daily test-drive quota of this account (`GET /api/test-drive`).
- The exact request of the model upload (method, content type) and the run-result fields: read from one real run.
