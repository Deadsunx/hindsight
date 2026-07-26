#!/usr/bin/env python3
"""The learning half of Hindsight — logistic regression, in pure Python.

Every model here is fit **walk-forward**: on any given run it trains only on
bets that have already settled, then predicts today. It never sees an outcome
it is being asked to predict. That discipline is what keeps the git timestamps
meaningful — a model trained on the future would make the whole archive a lie,
and no commit history could rescue it.

The two guarantees that matter, both enforced by tests:

  1. `features(days, date)` reads nothing after `date`. Truncating the archive
     at `date` must produce byte-identical features.
  2. `training_set()` excludes the day being predicted.

No dependencies, on purpose. This runs unattended for years; every import is a
future breakage, and at ~365 rows per predictor per year, logistic regression
is the right model class permanently — not a placeholder for something bigger.
"""

from __future__ import annotations

import math
from datetime import datetime

import predict

# Settled examples required before a model is allowed to place a bet at all.
# Below this the rule-based predictor carries the day on its own.
MIN_TRAIN = 25

# Models are never allowed to claim more certainty than this. Small samples
# produce absurd confidence otherwise, and overconfidence is the one failure
# the Brier score punishes hardest.
MAX_CONFIDENCE = 0.95


# --------------------------------------------------------------------------
# logistic regression
# --------------------------------------------------------------------------


def sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class Logistic:
    """Batch gradient descent with feature standardisation and L2.

    Deterministic: weights start at zero and the step count is fixed, so the
    same archive always produces the same model. That matters here — a
    prediction anyone can re-derive from the public record is worth more than
    one that depends on a random seed.
    """

    def __init__(self, l2: float = 1.0, lr: float = 0.5, epochs: int = 800):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w: list[float] = []
        self.b = 0.0
        self.mu: list[float] = []
        self.sd: list[float] = []

    def _standardise(self, row):
        return [(row[j] - self.mu[j]) / self.sd[j] for j in range(len(row))]

    def fit(self, X: list, y: list) -> "Logistic":
        n, d = len(X), len(X[0])
        self.mu = [sum(row[j] for row in X) / n for j in range(d)]
        self.sd = []
        for j in range(d):
            var = sum((row[j] - self.mu[j]) ** 2 for row in X) / n
            # A constant feature has zero spread; dividing by 1 leaves it at 0
            # after centring, which is exactly the right no-op.
            self.sd.append(math.sqrt(var) or 1.0)

        Z = [self._standardise(row) for row in X]
        self.w = [0.0] * d
        self.b = 0.0

        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                p = sigmoid(sum(self.w[j] * Z[i][j] for j in range(d)) + self.b)
                err = p - y[i]
                for j in range(d):
                    gw[j] += err * Z[i][j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j] / n)
            self.b -= self.lr * (gb / n)
        return self

    def predict_proba(self, row: list) -> float:
        z = sum(self.w[j] * v for j, v in enumerate(self._standardise(row))) + self.b
        return sigmoid(z)


# --------------------------------------------------------------------------
# small helpers over the archive
# --------------------------------------------------------------------------


def _mean(xs, default: float) -> float:
    return sum(xs) / len(xs) if xs else default


def _std(xs, default: float) -> float:
    if len(xs) < 2:
        return default
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) or default


def series(days: dict, date: str, key: str, field: str, n: int) -> list:
    """The last `n` values of observations[key][field], newest first.

    Walks strictly backwards from `date`, so it can never read the future.
    Missing or failed days are skipped rather than interpolated.
    """
    out = []
    cursor = date
    for _ in range(n):
        obs = predict.obs_of(days, cursor, key)
        if obs is not None and isinstance(obs.get(field), (int, float)):
            out.append(float(obs[field]))
        cursor = predict.shift(cursor, -1)
    return out


# --------------------------------------------------------------------------
# features — one builder per family, fixed length, no lookahead
# --------------------------------------------------------------------------
#
# Each returns a fixed-length list of floats, or None when the day lacks the
# minimum data. Fixed length matters: the model is refit from scratch daily,
# and a vector that changed width with archive depth would be untrainable.


def feat_btc(days, date):
    prices = series(days, date, "btc", "usd", 10)
    if len(prices) < 2:
        return None
    obs = predict.obs_of(days, date, "btc") or {}
    change = float(obs.get("usd_24h_change") or 0.0)
    back3 = prices[min(3, len(prices) - 1)]
    back7 = prices[min(7, len(prices) - 1)]
    rets = [(prices[i] / prices[i + 1] - 1) * 100 for i in range(len(prices) - 1)]
    return [
        change,
        (prices[0] / back3 - 1) * 100,
        (prices[0] / back7 - 1) * 100,
        _std(rets, 1.0),
    ]


def feat_kp(days, date):
    kps = series(days, date, "kp", "max_kp", 30)
    if not kps:
        return None
    today = kps[0]
    mean7 = _mean(kps[:7], today)
    # Solar activity recurs on a ~27-day rotation, so the value four weeks ago
    # carries real signal. Falls back to the weekly mean until the archive is
    # deep enough to have it.
    lag27 = kps[27] if len(kps) > 27 else mean7
    return [today, _mean(kps[:3], today), mean7, lag27]


def feat_quake(days, date):
    mags = series(days, date, "quakes", "max_mag", 14)
    counts = series(days, date, "quakes", "count", 14)
    if not mags:
        return None
    recent = mags[:7]
    return [
        mags[0],
        _mean(counts[:7], 120.0),
        _mean(recent, mags[0]),
        sum(1 for m in recent if m >= 5.0) / len(recent),
    ]


def feat_wiki(days, date):
    obs = predict.obs_of(days, date, "wiki")
    if not obs or not obs.get("top"):
        return None
    top = obs["top"]
    streak, cursor = 0, date
    for _ in range(30):  # bounded so an odd archive cannot spin here
        prev = predict.obs_of(days, cursor, "wiki")
        if prev and prev.get("top") == top:
            streak += 1
            cursor = predict.shift(cursor, -1)
        else:
            break
    views = series(days, date, "wiki", "views", 7)
    today_views = views[0] if views else 0.0
    mean_views = _mean(views, today_views or 1.0)
    return [
        float(min(streak, 10)),
        math.log1p(today_views),
        today_views / mean_views if mean_views else 1.0,
        1.0 if streak <= 1 else 0.0,
    ]


def feat_hn(days, date):
    obs = predict.obs_of(days, date, "hn")
    if not obs or not obs.get("top10"):
        return None
    top10 = list(obs["top10"])
    prev = predict.obs_of(days, predict.shift(date, -1), "hn")
    if prev and prev.get("top10"):
        overlap = len(set(top10) & set(prev["top10"])) / 10.0
        prev_top = prev.get("top_id")
        rank = float(top10.index(prev_top) + 1) if prev_top in top10 else 11.0
    else:
        overlap, rank = 0.2, 11.0
    # Weekend front pages churn differently from weekday ones.
    weekday = float(datetime.strptime(date, "%Y-%m-%d").weekday())
    return [overlap, rank, weekday]


def feat_forecast(days, date):
    obs = predict.obs_of(days, date, "weather")
    if not obs or obs.get("forecast_next") is None:
        return None
    today_high = obs.get("today")
    swing = abs(obs["forecast_next"] - today_high) if today_high is not None else 0.0
    actuals = series(days, date, "weather", "actual_prev", 10)

    # How wrong this forecaster has been lately — the feature that actually
    # carries signal. Each past forecast is matched to the completed high for
    # the date it was about.
    errors, cursor = [], date
    for _ in range(10):
        made = predict.obs_of(days, cursor, "weather")
        if made and made.get("forecast_next") is not None and made.get("forecast_next_date"):
            target = made["forecast_next_date"]
            for offset in (1, 2):
                settle_date = predict.shift(target, offset)
                # A forecast made on D is only settled on D+2, so the most
                # recent usable error is always a couple of days stale. Reading
                # past `date` here would leak the future into the model and
                # quietly inflate its apparent skill.
                if settle_date > date:
                    continue
                settled = predict.obs_of(days, settle_date, "weather")
                if (
                    settled
                    and settled.get("actual_prev_date") == target
                    and settled.get("actual_prev") is not None
                ):
                    errors.append(abs(settled["actual_prev"] - made["forecast_next"]))
                    break
        cursor = predict.shift(cursor, -1)

    return [swing, _std(actuals, 1.5), _mean(errors, 1.2), max(errors) if errors else 2.0]


FEATURES = {
    "btc_up": feat_btc,
    "kp_storm": feat_kp,
    "quake_m5": feat_quake,
    "wiki_hold": feat_wiki,
    "hn_fade": feat_hn,
    "forecast_hit": feat_forecast,
}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def training_set(family: str, days: dict, grades: list, today: str):
    """Settled (features, outcome) pairs for one family, strictly before today.

    Labels come from the same graded record the public ledger is built from —
    there is no second, parallel notion of truth that could drift from it.
    """
    featfn = FEATURES[family]
    X, y = [], []
    for grade in grades:
        if grade["id"] != family or grade["made_on"] >= today:
            continue
        row = featfn(days, grade["made_on"])
        if row is None:
            continue
        X.append(row)
        y.append(1.0 if grade["actual"] else 0.0)
    return X, y


def probability(family: str, days: dict, grades: list, today: str):
    """P(event happens) for today, or None if the model may not bet yet.

    Returns (probability, detail) so the archive can record what the model was
    working from — sample size, and whether it fell back.
    """
    X, y = training_set(family, days, grades, today)
    if len(X) < MIN_TRAIN:
        return None

    row = FEATURES[family](days, today)
    if row is None:
        return None

    positives = sum(y)
    if positives == 0 or positives == len(y):
        # One-sided history: gradient descent would chase an infinite weight.
        # Laplace smoothing gives an honest, bounded estimate instead.
        p = (positives + 1.0) / (len(y) + 2.0)
        return p, {"n_train": len(X), "fallback": "laplace", "features": row}

    model = Logistic().fit(X, y)
    p = model.predict_proba(row)
    return p, {
        "n_train": len(X),
        "fallback": None,
        "features": row,
        "weights": [round(w, 4) for w in model.w],
        "bias": round(model.b, 4),
    }


def call_and_confidence(p: float):
    """Turn P(event) into a call plus confidence in that call."""
    call = p >= 0.5
    confidence = p if call else 1.0 - p
    return call, round(min(MAX_CONFIDENCE, max(0.5, confidence)), 3)
