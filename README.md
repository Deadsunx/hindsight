# Hindsight

Six falsifiable forecasts a day, committed to public git **before the outcome
exists**, then scored against a hand-written rule, a coin flip, and the base
rate. Pure standard-library Python, no API keys, no dependencies.

**Live page → https://hindsight-deadsunx.vercel.app/**
_(mirrored at [deadsunx.github.io/hindsight](https://deadsunx.github.io/hindsight/))_

## The leak the test caught

The model half is a walk-forward logistic regression: on any run it trains only
on bets that have already settled, then predicts today. That claim is worth
nothing unless something enforces it, so the enforcement is a test rather than a
comment.

`test_features_unchanged_when_future_is_corrupted` builds a 60-day archive,
picks a target date, and records every feature vector for that date. It then
**overwrites every day after the target with garbage** — Bitcoin at $1, Kp at
99, every Wikipedia headline replaced — and recomputes. If any feature moves by
so much as a float, the feature read the future and the test fails.

It failed the first time it ran. The forecast-error feature walked back through
past forecasts and matched each to the completed high for the date it was about
— but a forecast made on day *D* only settles on *D+2*, and the loop was
reading the settling day without checking it was still in the past. On a real
run that day had not happened yet; on a backfill it had. The feature was reading
two days ahead of itself.

That is a leak that shows up as *better* scores, never as a crash. It would have
quietly inflated the model's apparent skill for as long as the project ran, and
every git timestamp would still have been perfectly honest. The fix is a
two-line guard in [`learn.py`](learn.py); the reason it was ever found is the
test.

<!-- LEDGER:START -->
### Standing — 2026-08-12

**78 right / 101 settled (77%)** · Brier 0.145 · log loss 0.449 · 7 still open

Saying "50%" to all 101 of them instead would score Brier 0.250, log loss 0.693. Lower is better; every figure below is only worth what it beats.

| Call | Answered by | Settled | Right | Brier | Log loss |
| --- | --- | --- | --- | --- | --- |
| Bitcoin will be higher tomorrow than it is today. | momentum | n=17 | 8/17 (47%) | 0.255 | 0.703 |
| Geomagnetic Kp will exceed 3 within the next 24 hours. | persistence | n=17 | 12/17 (71%) | 0.206 | 0.603 |
| A magnitude 5.0+ earthquake will strike somewhere tomorrow. | base rate | n=17 | 17/17 (100%) | 0.007 | 0.077 |
| Today's most-read Wikipedia article will still be #1 tomorrow. | persistence | n=17 | 11/17 (65%) | 0.219 | 0.628 |
| The current #1 story on Hacker News will fall out of the top 10. | decay | n=17 | 15/17 (88%) | 0.104 | 0.362 |
| Tomorrow's high in New Delhi will land within 2°C of today's forecast. | trust the forecaster | n=16 | 15/16 (94%) | 0.077 | 0.310 |
| _coin flip — the baseline_ | _50% to everything_ | n=101 | _50%_ | _0.250_ | _0.693_ |

_Sample is 101 settled bets across 18 days. Nothing here is significant yet, and it is published daily precisely so that it becomes so._

**Open bets, placed today:**

- **no** — Bitcoin will be higher tomorrow than it is today. _(52% confident, momentum)_
- **yes** — Geomagnetic Kp will exceed 3 within the next 24 hours. _(70% confident, persistence)_
- **yes** — A magnitude 5.0+ earthquake will strike somewhere tomorrow. _(95% confident, base rate)_
- **no** — Today's most-read Wikipedia article will still be #1 tomorrow. _(62% confident, persistence)_
- **yes** — The current #1 story on Hacker News will fall out of the top 10. _(88% confident, decay)_
- **yes** — Tomorrow's high in New Delhi will land within 2°C of today's forecast. _(80% confident, trust the forecaster)_

<!-- LEDGER:END -->

That block is rewritten by the daily run, not by hand.

## What it bets on

| Call | Source | Rule |
| ---- | ------ | ---- |
| Bitcoin closes higher tomorrow | CoinGecko | momentum |
| Geomagnetic Kp exceeds 3 within 24 h | NOAA SWPC | persistence |
| A M5.0+ earthquake strikes somewhere | USGS | base rate |
| Today's top Wikipedia article stays #1 | Wikimedia pageviews | persistence |
| The #1 Hacker News story falls out of the top 10 | HN Firebase | decay |
| Tomorrow's high lands within 2 °C of the forecast | Open-Meteo | trust the forecaster |

The last one is not a weather prediction — it is a **running scorecard on the
weather service**, which nobody publishes.

<!-- VERDICT:START -->
## Where it does not beat the baseline

Judged against the coin flip's Brier of 0.250, over 101 settled bets:

- **Bitcoin will be higher tomorrow than it is today.** scores Brier **0.255** against the coin flip's 0.250 over n=17. It is worse than shrugging.
- **A magnitude 5.0+ earthquake will strike somewhere tomorrow.** is effectively one-sided: the event happened 100% of the time, so always calling the majority side would have scored 100% against this rule's 100%. Accuracy here measures the question, not the predictor.
- **The current #1 story on Hacker News will fall out of the top 10.** is effectively one-sided: the event happened 88% of the time, so always calling the majority side would have scored 88% against this rule's 88%. Accuracy here measures the question, not the predictor.
- **Tomorrow's high in New Delhi will land within 2°C of today's forecast.** is effectively one-sided: the event happened 94% of the time, so always calling the majority side would have scored 94% against this rule's 94%. Accuracy here measures the question, not the predictor.

The model half has placed **zero** settled bets: it may not bet until 25 examples in its family have settled. Until then the head-to-head table is honestly blank rather than quietly filled with the rule's numbers.

_Derived from the 101 settled bets in the ledger on every run, not written by hand. At this sample size none of it is significant; it is published daily so that one day it might be._

<!-- VERDICT:END -->

## How the score is kept

Accuracy alone is close to useless here, so four figures are reported and each
carries its own sample size:

- **Accuracy** — how often the call was right. Cannot tell a cautious miss from
  a reckless one, and is trivially inflated by a one-sided question.
- **Brier score** — mean squared error of the probability assigned to the event.
  `0.250` is what "50% to everything" earns. Lower is better; higher means
  confidently wrong.
- **Log loss** — the same idea, punishing overconfidence far harder. A 95% call
  that misses costs about 3.0; a 55% miss costs about 0.8. Reported because
  Brier is comparatively forgiving of exactly the failure worth catching.
- **Calibration** — when it says 70%, is it right 70% of the time? Binned into
  confidence bands and plotted against the diagonal on the live page. This is
  the figure that needs months, which is the whole reason the run is daily.

Every one of them sits next to two references: the **coin flip** (0.250 Brier,
0.693 log loss, by construction) and the **base rate** — how often the event
simply happens, and what always calling the majority side would have scored.

### Honest bookkeeping

- **Open** — placed, not yet due.
- **Void** — the day that would have settled it never got archived. Counted
  separately and **never** counted as a win.
- **Shared days only** — the rule and the model are compared solely on days
  where both actually bet. Comparing lifetime accuracies would compare
  different sets of days and flatter whichever drew the easier ones.

## The model, and why it can't cheat

Each question is also answered by a logistic regression in [`learn.py`](learn.py)
— pure Python, no dependencies. Two properties make its record trustworthy, and
both are enforced by tests rather than by good intentions:

1. **Walk-forward only.** It trains solely on bets that have already settled,
   then predicts today. It never sees an outcome it is being asked to predict.
2. **Features are blind to the future.** `features(days, date)` reads nothing
   after `date` — the property the corruption test above exists to enforce.

The model stays out entirely until **25 settled examples** exist, and never
claims more than **95%** confidence. If a family's history is one-sided — as
`hn_fade` tends to be — it falls back to Laplace smoothing instead of letting
gradient descent chase an infinite weight into false certainty.

**The rules are never retired.** They are the control. A model that beats a
five-line heuristic by 0.02 Brier is a more interesting public result than one
claiming 90% accuracy with nothing to compare against.

## How it works

1. `.github/workflows/daily.yml` runs [`predict.py`](predict.py) twice a day.
2. **Observe** — six keyless sources, each wrapped in try/except. A dead API
   just means no call from that rule today; the commit still happens.
3. **Predict** — each rule makes one binary call with a stated confidence,
   using only what was knowable at that moment.
4. **Grade** — every past prediction is re-derived from the archive and settled
   if the truth has arrived.
5. It commits and pushes **as me**, so it counts toward my contribution graph.

Grading is a **pure function of the archive**, recomputed from scratch on every
run rather than accumulated. A bad day can never corrupt the ledger, and
`archive/ledger.json` can always be regenerated by re-running the script.

## Don't trust it — check it

```bash
python verify.py
```

No network, no dependencies. It answers the two questions the whole project
rests on:

1. **Is the scorecard honest arithmetic?** It recomputes `ledger.json` and
   `index.json` from `archive/` alone and compares. Hand-edit a total and it
   names the exact field:

   ```
   [FAIL] ledger.json does not match a fresh recomputation:
       .totals.accuracy: type NoneType != float
       .totals.correct: 0 != 3
   ```

2. **Did the bets really predate their outcomes?** It asks git when each
   archive day was last committed and requires that to be before the day it
   could possibly be settled. Backdate a prediction and it fails.

Exit code 0 means verified. CI runs it on every push, and the daily workflow
runs it *before* pushing — a record that does not reconcile is withheld rather
than published.

What it deliberately does **not** prove: that the observations were accurate.
Nobody can re-fetch a past day's Hacker News front page. What it proves is that
nothing was altered afterwards — which is the property the archive exists to
have.

## Tests

```bash
python -m unittest test_predict test_learn test_verify -v
```

Offline, no network, no dependencies. **[TESTING.md](TESTING.md)** walks through
what each test enforces and why — in particular the leak test, what it corrupts,
what it asserts, and the bug it caught.

## Files

| Path | What |
| ---- | ---- |
| `predict.py` | the engine — observe, predict, grade, score |
| `learn.py` | logistic regression + walk-forward features |
| `verify.py` | audits the published record; no network needed |
| `test_*.py` | offline tests, no network |
| `TESTING.md` | what each test enforces, and why |
| `archive/YYYY-MM-DD.json` | one day: observations + the bets placed that day |
| `archive/ledger.json` | the full scorecard the page reads |
| `archive/index.json` | per-day summary for the calendar |
| `index.html` | the page |

## Run it locally

```bash
python predict.py
```

Only the Python standard library is used — nothing to install, no API keys.

## Move the forecast bet to your city

Set repo variables (Settings → Secrets and variables → Actions → Variables),
or env vars locally:

```bash
CITY_NAME="Bamako" CITY_LAT=12.6392 CITY_LON=-8.0029 CITY_TZ=Africa/Bamako python predict.py
```
