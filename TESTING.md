# Testing

```bash
python -m unittest test_predict test_learn test_verify -v
```

75 tests, no network, no dependencies, under half a second. CI runs them on
every push, and the daily workflow runs `verify.py` *before* it pushes — a
record that does not reconcile is withheld rather than published.

The suite is organised around one idea: **the parts that can lie are tested
hardest.** A dead API is allowed to fail, because a missing observation just
means no bet from that source today and the archive records the gap honestly.
Grading, feature construction and scoring are not allowed to fail, because if
any of them is wrong the public track record is a lie that looks exactly like a
good one.

---

## The leak test

**`test_learn.py :: TestNoLookahead`** — the single most important test here,
and the reason the file exists.

### What it does

`test_features_unchanged_when_future_is_corrupted` is the strong form:

1. Build a 60-day synthetic archive with realistic shape — prices that drift,
   temperatures that wander, Kp values, Wikipedia titles that sometimes repeat.
2. Pick a target date 30 days in.
3. Record the feature vector every family produces **for that date**.
4. **Overwrite every single day after the target with garbage** — Bitcoin at
   $1.00, `usd_24h_change` at 999, Kp at 99, magnitude 9.9, every Wikipedia
   headline replaced with `"Corrupted"`, every HN id set to 1, every temperature
   99°C.
5. Recompute the same feature vectors.
6. Assert they are **identical**.

A weaker sibling, `test_features_ignore_future_days`, truncates the archive at
the target instead of corrupting it and asserts the same equality. Truncation
catches a feature that *reads* the future; corruption also catches one that
reads it and happens to be insensitive to the particular values in a fixture.

`test_training_set_excludes_the_day_being_predicted` covers the other half:
labels, not features. The training set for day *D* must contain nothing made on
or after *D*.

### What it caught

It failed the first time it ran, on `feat_forecast`.

That feature measures how wrong the weather service has been lately — the only
feature in the family carrying real signal. To compute it, it walks back through
past forecasts and matches each to the completed high for the date it was about.

The problem is the settling delay. A forecast made on day *D* is *about* day
*D+1*, and the completed high for *D+1* only appears in the observation taken on
*D+2*. The loop looked up that settling day without checking it was still in the
past. On a live run the day simply did not exist yet, so nothing came back and
the bug was invisible. On any archive deep enough to have it — which is to say,
during the corruption test — it read two days ahead of itself.

The fix is the `if settle_date > date: continue` guard in
[`learn.py`](learn.py), inside `feat_forecast`.

### Why it matters more than it looks

This class of bug **never announces itself.** It does not crash, it does not
raise, it does not produce an implausible number. It produces *better scores*.
Every git commit timestamp would have remained perfectly honest, `verify.py`
would have passed every audit, and the model would have shown a quiet, durable
edge over the rule baseline that was entirely an artefact of reading answers it
should not have had.

The tamper-evidence machinery in `verify.py` is useless against it, because
nothing was tampered with. Only a test that actively corrupts the future and
demands byte-identical output can find it.

---

## What the rest of the suite enforces

### `test_predict.py` — grading and scoring

| Group | Enforces |
| --- | --- |
| `TestEnv` | An unset GitHub Actions repo variable arrives as `""`, not as absent. A plain `.get()` would build a URL with a blank latitude that quietly 400s — which is how the first CI run lost the weather bet. |
| `TestResolvers` | Each question settles in **both** directions, and thresholds are exactly where they claim: Kp is strict (`> 3`), quake magnitude is inclusive (`>= 5.0`). A forecast settles from *any* later run carrying the completed high, so a missed cron day does not void it — but a run carrying the wrong date does not settle it either. |
| `TestGrading` | Grading is **idempotent** and a pure function of the archive. Unsettled bets are absent rather than guessed at; retired predictor ids are skipped rather than crashing. |
| `TestLedger` | Accuracy, Brier, calibration bucketing, and the open-versus-void split. An empty archive must not divide by zero, and every predictor row carries its own sample size. |
| `TestScoring` | The metrics themselves — see below. |
| `TestPredictors` | Every rule declares a question, a horizon and a resolver; confidence never drops below 0.5 or above 0.95; every family has both a rule and a model variant that **share a resolver**, so the two are provably answering the same question. |

`TestScoring` deserves calling out, because it is what makes the published
numbers meaningful rather than decorative:

- `test_p_event_inverts_a_no_call` — confidence is stated in the *call*, not the
  event. A 70% "no" is a 30% chance of the event. Every scoring rule has to
  convert first, and getting this backwards would silently invert half the
  Brier score.
- `test_log_loss_punishes_the_confident_miss_harder_than_brier` — the reason log
  loss is reported at all. A 55% miss and a 95% miss are *identical* to
  accuracy. The test asserts log loss opens a strictly wider gap between them
  than Brier does.
- `test_a_certain_wrong_call_does_not_produce_infinity` — log loss is unbounded;
  one 100%-confident miss would poison the running average forever. Clamped, and
  the clamp is tested.
- `test_base_rate_exposes_an_unpredictable_question` — a question whose event
  fires nine days in ten hands you 90% accuracy for free. Without the base-rate
  column there is no way to tell that apart from skill.
- `test_empty_scores_are_none_not_zero` — zero would render as a *perfect* Brier
  score. `None` renders as "no evidence". This distinction is the difference
  between an honest empty scoreboard and a fraudulent one.

### `test_learn.py` — the model

Beyond the leak tests: the logistic regression learns a separable signal, is
**deterministic** (zero-initialised weights, fixed epoch count — so anyone can
re-derive the same prediction from the public archive), survives a constant
feature without dividing by zero, keeps probabilities in `[0, 1]`, and has a
numerically stable sigmoid at ±800.

`TestColdStart` covers the refusals, which matter more than the predictions:
the model does not bet below 25 settled examples; a one-sided history falls back
to Laplace smoothing rather than letting gradient descent chase an infinite
weight into false certainty; confidence is bounded to `[0.5, 0.95]`.

`TestHeadToHead` enforces that rule and model are compared on **shared days
only** — days where both actually bet — and that predictions archived before the
variant system existed are skipped rather than crashing the comparison.

### `test_verify.py` — the auditor

A verifier that cannot fail is decoration, so most of these tests tamper with
something and assert the audit notices: inflating the correct count, deleting a
losing grade, flipping an outcome, and rewriting a past observation to turn a
loss into a win. Each one must produce a diff naming the exact field.

`TestNormalise` covers the opposite failure — the audit must *not* fire on
`updated` timestamps or on question text rebuilt from a different `CITY_NAME`,
or a verifier running in another city would see a spurious mismatch on numbers
that are actually identical.

---

## Regenerating the ledger

`ledger.json` and `index.json` are committed, and `verify.py` requires them to
match a fresh recomputation from `archive/` exactly. Change anything about how
they are built and the committed copies must be regenerated in the same commit,
or CI fails and the daily run withholds its push.

Running `python predict.py` does this, but it also hits the network and places a
real bet. To rebuild the derived files from the existing archive alone:

```python
import predict

days = predict.load_days()
today = max(days)                      # what verify.py assumes
grades = predict.grade_all(days)
ledger = predict.build_ledger(days, grades, today)

predict._write(predict.ARCHIVE / "ledger.json", ledger)
predict._write(predict.ARCHIVE / "index.json", predict.build_index(days, grades))
```

Then confirm with `python verify.py`, which should print `VERIFIED` and exit 0.
