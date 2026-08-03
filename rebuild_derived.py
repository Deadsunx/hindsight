#!/usr/bin/env python3
"""Rebuild the derived files from the archive. No network, no new bets.

`ledger.json` and `index.json` are pure functions of the archived day files,
so they can always be regenerated without touching the record itself. This is
what to run when the derived files fall out of step with the archive:

  * after a rebase that brought in an archived day from another run
  * after a change to how the ledger is computed

It deliberately does **not** call any observer or place a prediction. Running
it is always safe; it can only ever rewrite files that were already derivable
from what is committed.

    python rebuild_derived.py
"""

import sys

import predict


def main() -> int:
    days = predict.load_days()
    if not days:
        print("archive is empty; nothing to rebuild")
        return 1

    today = max(days)
    grades = predict.grade_all(days)
    predict._write(predict.ARCHIVE / "ledger.json", predict.build_ledger(days, grades, today))
    predict._write(predict.ARCHIVE / "index.json", predict.build_index(days, grades))
    print(f"rebuilt from {len(days)} archived day(s); newest {today}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
