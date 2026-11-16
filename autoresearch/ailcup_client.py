"""Client for the real AIL Cup Garage (https://ailcup.ail-lab.fiit.stuba.sk).

The API is undocumented; it was read from the site's own JavaScript (see docs/ail_cup_api.md).
Login comes from environment variables, never from a file:
    set AILCUP_LOGIN=...      set AILCUP_PASSWORD=...

Usage (one real test drive of a controller + model):
    venv\\Scripts\\python -m autoresearch.ailcup_client ail_cup\\simple_controller.py ^
        --model ail_cup\\wm2_c4_value_garage.npz --laps 2 [--track <id>] [--label baseline]
    venv\\Scripts\\python -m autoresearch.ailcup_client --quota     (test drives left today)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://ailcup.ail-lab.fiit.stuba.sk"
CSRF_HEADER = {"x-ailcup-csrf": "1"}   # the site rejects every call without it
REQUEST_TIMEOUT_S = 30
UPLOAD_TIMEOUT_S = 300                  # models up to 50 MB
POLL_EVERY_S = 6                        # the site's own page polls every 6 s
WAIT_MAX_S = 30 * 60
MAX_CODE_BYTES = 64 * 1024              # the editor refuses bigger files
MAX_MODEL_BYTES = 50 * 1024 * 1024
FINAL_STATES = ("done", "error")


class AilCupError(RuntimeError):
    """A call to the Garage failed; the message is the site's own `detail` when it gives one."""


class AilCupClient:
    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = requests.Session()
        self.http.headers.update(CSRF_HEADER)

    # ---------- low level
    def _call(self, method: str, path: str, *, json_body=None, data=None, headers=None,
              timeout: float = REQUEST_TIMEOUT_S):
        try:
            resp = self.http.request(method, self.base_url + path, json=json_body, data=data,
                                     headers=headers, timeout=timeout)
        except requests.RequestException as e:
            raise AilCupError(f"{method} {path}: network error: {e}") from e
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if not resp.ok:
            detail = body.get("detail") if isinstance(body, dict) else None
            raise AilCupError(f"{method} {path}: HTTP {resp.status_code}: {detail or resp.text[:200]}")
        return body

    # ---------- account
    def login(self, login: str | None = None, password: str | None = None) -> dict:
        login = login or os.environ.get("AILCUP_LOGIN")
        password = password or os.environ.get("AILCUP_PASSWORD")
        if not login or not password:
            raise AilCupError("set AILCUP_LOGIN and AILCUP_PASSWORD first")
        self._call("POST", "/api/auth/login", json_body={"login": login, "password": password})
        return self._call("GET", "/api/auth/me")

    def _test_drive_page(self) -> dict:
        page = self._call("GET", "/api/test-drive")
        if isinstance(page, list):          # the JS reads {runs: [...]}, but accept a bare list too
            return {"runs": page}
        if not isinstance(page, dict):
            raise AilCupError(f"GET /api/test-drive: unexpected reply: {str(page)[:200]}")
        return page

    def quota(self) -> dict:
        """The test-drive list; its quota fields (per_day / left / total) say how many drives are left."""
        page = self._test_drive_page()
        summary = {k: v for k, v in page.items() if not isinstance(v, (list, dict))}   # per_day, left, total, more
        summary["big_fields"] = {k: f"{type(v).__name__} of {len(v)}" for k, v in page.items()
                                 if isinstance(v, (list, dict))}                      # runs, tracks: see --save-page
        return summary

    def save_page(self, path: Path) -> None:
        """The whole /api/test-drive page (runs, track maps) as JSON, to study the real tracks."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._test_drive_page(), indent=1), encoding="utf-8")

    # ---------- garage
    def check_code(self, code: str) -> dict:
        return self._call("POST", "/api/code/check", json_body={"code": code})

    def save_version(self, code: str, label: str) -> dict:
        return self._call("POST", "/api/submissions", json_body={"label": label, "code": code, "folder_id": None})

    def attach_model(self, submission_id: int, model_bytes: bytes, name: str) -> dict:
        return self._call("PUT", f"/api/model-upload/{submission_id}/{requests.utils.quote(name, safe='')}",
                          data=model_bytes,
                          headers={"content-type": "application/octet-stream"}, timeout=UPLOAD_TIMEOUT_S)

    def start_test_drive(self, code: str, laps: int, track: str | None = None,
                         model_from: int | None = None) -> int:
        body: dict = {"code": code, "laps": laps}
        if track:
            body["track"] = track
        if model_from is not None:
            body["model_from"] = model_from
        return _field(self._call("POST", "/api/test-drive", json_body=body), "job_id", "start test drive")

    def find_run(self, job_id: int) -> dict | None:
        runs = self._test_drive_page().get("runs") or []
        return next((r for r in runs if isinstance(r, dict) and r.get("id") == job_id), None)

    def wait_run(self, job_id: int, wait_max_s: float = WAIT_MAX_S) -> dict:
        deadline = time.monotonic() + wait_max_s
        while time.monotonic() < deadline:
            run = self.find_run(job_id)
            if run and run.get("status") in FINAL_STATES:
                return run
            time.sleep(POLL_EVERY_S)
        raise AilCupError(f"run {job_id} not finished after {wait_max_s:.0f} s")

    # ---------- the whole flow
    def test_drive(self, code_path: Path, model_path: Path | None, laps: int,
                   track: str | None, label: str) -> dict:
        code, model_bytes = read_inputs(code_path, model_path)   # local problems fail before any network call
        check = self.check_code(code)
        submission_id = None
        if model_path is not None:
            submission_id = _field(self.save_version(code, label), "id", "save version")
            self.attach_model(submission_id, model_bytes, model_path.stem)
        job_id = self.start_test_drive(code, laps, track, submission_id)
        run = self.wait_run(job_id)
        return to_row(run, label, check)


def _field(body, key: str, step: str):
    """body[key], or a clear AilCupError when the site's reply has another shape than its JavaScript suggests."""
    if isinstance(body, dict) and key in body:
        return body[key]
    raise AilCupError(f"{step}: no '{key}' in the reply: {str(body)[:200]}")


def read_inputs(code_path: Path, model_path: Path | None) -> tuple[str, bytes | None]:
    """Read and size-check the controller and the model on this PC."""
    try:
        code = code_path.read_text(encoding="utf-8")
        model_bytes = model_path.read_bytes() if model_path is not None else None
    except OSError as e:
        raise AilCupError(f"cannot read {e.filename}: {e.strerror}") from e
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise AilCupError(f"{code_path.name} is larger than 64 KB")
    if model_bytes is not None and len(model_bytes) > MAX_MODEL_BYTES:
        raise AilCupError(f"{model_path.name} is {len(model_bytes) / 1e6:.1f} MB, the Garage allows 50 MB")
    return code, model_bytes


def to_row(run: dict, label: str, check: dict) -> dict:
    """One finished run -> one flat row for results.tsv / score.py."""
    res = run.get("result") or {}
    finished = res.get("status") == "finished"
    latency = res.get("latency_ms") or {}
    return {
        "label": label,
        "run_id": run.get("id"),
        "sim_error": run.get("error") if run.get("status") == "error" else None,  # the platform failed: retry, do not score
        "finished": finished,
        "total_time_s": res.get("total_time") if finished else None,  # the penalty is score.py's job
        "dnf_reason": None if finished else res.get("dnf_reason"),
        "dnf_detail": None if finished else res.get("dnf_detail"),
        "energy_wh": res.get("energy_wh"),
        "step_p95_ms": latency.get("p95"),
        "code_check": check,
        "driver_log_tail": (run.get("driver_log") or "")[-1500:],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One real AIL Cup Garage test drive")
    ap.add_argument("controller", nargs="?", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--track")
    ap.add_argument("--label", default="autoresearch")
    ap.add_argument("--quota", action="store_true", help="only print the test drives left today")
    ap.add_argument("--save-page", type=Path, help="save the whole test-drive page (runs + track maps) as JSON")
    args = ap.parse_args(argv)

    client = AilCupClient()
    try:
        if args.controller is not None and not args.quota:
            read_inputs(args.controller, args.model)              # wrong paths fail before the login
        me = client.login()
        print(f"logged in as {me.get('handle', '?')}", file=sys.stderr)
        if args.save_page:
            client.save_page(args.save_page)
            print(f"saved {args.save_page}", file=sys.stderr)
            return 0
        if args.quota or args.controller is None:
            print(json.dumps(client.quota(), indent=2))
            return 0
        row = client.test_drive(args.controller, args.model, args.laps, args.track, args.label)
    except AilCupError as e:
        print(f"AIL Cup: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("AIL Cup: stopped by the user (a started run may still be running on the site)", file=sys.stderr)
        return 130
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
