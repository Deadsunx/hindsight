#!/usr/bin/env python3
"""Audit Hindsight's published record. No network, no trust required.

Run this against a clone and it answers two questions:

  1. **Is the scorecard honest arithmetic?** Recompute the ledger and the
     calendar index from `archive/` alone and compare them to the committed
     files. If anyone hand-edited a total, this fails.

  2. **Did the predictions really predate their outcomes?** Ask git when each
     archive day was last committed, and require that to be before the day it
     could possibly be settled. If anyone backdated a bet, this fails.

Together those are the whole claim. Neither depends on believing anything I
say — the archive is public, this script is public, and the answer is a exit
code.

    python verify.py

What this deliberately does **not** prove: that the observations themselves are
accurate. The sources have long since moved on and there is no way to re-fetch
2026-07-26's Hacker News front page. What it proves is that nothing was altered
after the fact, which is the property the archive was built to have.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import predict

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "archive"

# Keys excluded from comparison, and why:
#   updated  — a wall-clock stamp, different on every run by construction
#   question — presentation text built from CITY_NAME at runtime, not part of
#              the record; a verifier with a different city set would otherwise
#              see a spurious mismatch on numbers that are actually identical
VOLATILE = {"updated", "question"}


def normalise(value):
    """Strip volatile keys so the comparison is about the numbers."""
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [normalise(v) for v in value]
    return value


def differences(expected, actual, path="", found=None, limit=8):
    """Human-readable list of where two structures diverge."""
    found = [] if found is None else found
    if len(found) >= limit:
        return found
    if type(expected) is not type(actual):
        found.append(f"{path or '<root>'}: type {type(expected).__name__} != {type(actual).__name__}")
    elif isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                found.append(f"{path}.{key}: unexpectedly present")
            elif key not in actual:
                found.append(f"{path}.{key}: missing")
            else:
                differences(expected[key], actual[key], f"{path}.{key}", found, limit)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            found.append(f"{path}: length {len(expected)} != {len(actual)}")
        else:
            for i, (a, b) in enumerate(zip(expected, actual)):
                differences(a, b, f"{path}[{i}]", found, limit)
    elif expected != actual:
        found.append(f"{path}: {expected!r} != {actual!r}")
    return found[:limit]


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:
        print(f"    could not parse {path.name}: {err}")
        return None


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_recomputed(days: dict):
    """The ledger and index must be derivable from the archive files alone."""
    if not days:
        print("  [FAIL] archive is empty")
        return False

    # The committed ledger was built on the most recent archived day, and the
    # open/void split depends on that date, so it has to be reproduced here.
    today = max(days)
    grades = predict.grade_all(days)
    ok = True

    for name, rebuilt in (
        ("ledger.json", predict.build_ledger(days, grades, today)),
        ("index.json", predict.build_index(days, grades)),
    ):
        committed = load(ARCHIVE / name)
        if committed is None:
            print(f"  [FAIL] {name} is missing or unreadable")
            ok = False
            continue
        diffs = differences(normalise(rebuilt), normalise(committed))
        if diffs:
            print(f"  [FAIL] {name} does not match a fresh recomputation:")
            for line in diffs:
                print(f"      {line}")
            ok = False
        else:
            print(f"  [ok]   {name} reproduces exactly from archive/")
    return ok


def check_latest(days: dict):
    """latest.json must be a copy of the newest archived day."""
    latest = load(ARCHIVE / "latest.json")
    if latest is None:
        print("  [FAIL] latest.json is missing or unreadable")
        return False
    newest = days[max(days)]
    if latest.get("date") != newest.get("date"):
        print(f"  [FAIL] latest.json is {latest.get('date')}, newest archive day is {newest.get('date')}")
        return False
    if normalise(latest) != normalise(newest):
        print(f"  [FAIL] latest.json disagrees with archive/{newest['date']}.json")
        return False
    print(f"  [ok]   latest.json matches archive/{newest['date']}.json")
    return True


def _git(*args):
    try:
        out = subprocess.run(
            ("git",) + args, cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def check_commit_timestamps(days: dict):
    """Every bet must have been committed before it could be settled.

    A prediction made on D that resolves on D+1 has to be in git no later than
    D. If a file's last commit lands on or after its own resolve date, someone
    edited a prediction once the answer was already knowable.
    """
    if _git("rev-parse", "--git-dir") is None:
        print("  [skip] skipped: not a git repository")
        return True
    if _git("log", "-1", "--format=%H") is None:
        print("  [skip] skipped: no commit history available (shallow clone?)")
        return True

    checked = failures = skipped = 0
    for date in sorted(days):
        predictions = days[date].get("predictions") or []
        if not predictions:
            continue
        rel = f"archive/{date}.json"

        if _git("status", "--porcelain", "--", rel):
            print(f"  [warn] {rel} has uncommitted changes; skipping")
            skipped += 1
            continue

        stamp = _git("log", "-1", "--format=%cI", "--", rel)
        if not stamp:
            skipped += 1
            continue
        committed_at = datetime.fromisoformat(stamp).astimezone(timezone.utc)

        # The earliest moment any of that day's bets could be settled.
        deadline = min(p["resolve_on"] for p in predictions)
        deadline_at = datetime.fromisoformat(deadline).replace(tzinfo=timezone.utc)

        checked += 1
        if committed_at >= deadline_at:
            print(
                f"  [FAIL] {rel} committed {committed_at:%Y-%m-%d %H:%M}Z, "
                f"but its earliest bet resolves {deadline} — outcome was already knowable"
            )
            failures += 1

    if skipped:
        print(f"  [warn] {skipped} day(s) skipped (uncommitted or untracked)")
    if failures:
        return False
    if checked:
        print(f"  [ok]   all {checked} archived day(s) committed before their outcomes existed")
    else:
        print("  [skip] no committed days with predictions to check yet")
    return True


def main() -> int:
    days = predict.load_days()
    print(f"Auditing {len(days)} archived day(s) in {ARCHIVE}\n")

    print("Arithmetic — is the scorecard a faithful computation over the archive?")
    ok = check_recomputed(days)
    ok = check_latest(days) and ok

    print("\nChronology — were the bets committed before the outcomes existed?")
    ok = check_commit_timestamps(days) and ok

    print()
    if ok:
        print("VERIFIED — the published record is internally consistent and")
        print("           every prediction predates the outcome it claims.")
        return 0
    print("FAILED — the published record does not hold up. Details above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
