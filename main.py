"""Launch-day Twitch CCU forecaster for make.fun's gta6-twitch-ccu-forecast-01.

The model, in one paragraph: an hour of a launch day is predicted from the
launch days of the comparable launches the case itself ships. Every comparable
contributes its release-day curve as log-ratios to its own hour 1,
lr[h] = ln(viewers[h]) - ln(viewers[1]). For each hour h the entry takes the
weighted quantiles of those log-ratios across comparables, where a comparable's
weight falls off with the distance between its hour-1 audience and this case's
hour-1 print (a Gaussian kernel on log scale, bandwidth 0.65) and is boosted
when its unlock pattern matches the target's (rolling midnight-local unlocks,
read from the unlock/notes text). The hour's five quantiles blend the empirical
weighted quantile with a normal fit (median and a robust sigma from the
weighted 16th/84th percentiles, both smoothed across neighbouring hours), the
log-ratios are shrunk 10% toward flat persistence, and the spread is scaled by
0.95. The result is anchored on the case's hour-1 print p:
q_tau(h) = p * exp(log_ratio_tau(h)).

Why this shape: the contest scores pinball loss on the log scale, which the
true quantiles minimise, and launch days differ mostly by when their audience
arrives (regional unlock waves), not by whether it arrives. Conditioning on
hour-1 scale and unlock pattern picks the comparable days whose wave structure
resembles this launch; the blend and the shrinkage keep the tails honest where
only a few dozen comparable days inform them. Hours a case's comparables say
nothing about fall back to the persistence spread (flat at p, widening like
0.15 * sqrt(h - 1) in log space), so the output is always well formed.

The program reads every *.in case file in the input directory (the grader runs
one case at a time) and writes one <id>.out per case: {"hours": [{"hour": h,
"q": [q05, q25, q50, q75, q95]} ...]} for hours 2..24. Standard library only,
no network, no randomness: the same input bytes give the same output bytes.

Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Model hyperparameters, tuned by leave-one-out backtests over the comparable
# launches the practice cases ship (see README).
SCALE_BANDWIDTH = 0.65   # Gaussian kernel width on ln(hour-1 viewers)
ROLLING_BONUS = 1.5      # weight multiplier when the unlock pattern matches
SHRINK = 0.9             # log-ratios are multiplied by this (toward flat)
GAMMA = 0.95             # spread scale around the median
MIN_DAY_POINTS = 20      # a comparable needs at least this many release-day hours
FALLBACK_SIGMA = 0.15    # persistence spread per sqrt hour, as the baseline uses

_NORM = NormalDist()


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def round2(x: float) -> float | int:
    """Round half up to 2 decimals; whole numbers come back as int."""
    value = math.floor(x * 100 + 0.5) / 100
    return int(value) if float(value).is_integer() else value


def is_rolling(text: str) -> bool:
    """True when a launch's unlock text describes rolling midnight-local unlocks."""
    t = (text or "").lower()
    return "midnight local" in t or ("midnight" in t and "rolling" in t)


def day_curve(comparable: dict) -> dict[int, float]:
    """The comparable's release-day hourly viewers, {hour 1..24: viewers}."""
    start = parse_ts(comparable["day_start"])
    day: dict[int, float] = {}
    for point in comparable.get("hourly", []):
        t = parse_ts(point["from"])
        if start <= t < start + timedelta(hours=24):
            hour = int((t - start).total_seconds() // 3600) + 1
            day[hour] = point["viewers"]
    return day


def build_pool(case: dict) -> list[dict]:
    """Comparable launches with a usable release day, as log-ratio sources."""
    pool = []
    for comparable in case.get("comparables", []):
        day = day_curve(comparable)
        first = day.get(1)
        if len(day) < MIN_DAY_POINTS or not first or first <= 0:
            continue
        pool.append({
            "log_first": math.log(first),
            "rolling": is_rolling(comparable.get("unlock", "")),
            "log_ratio": {h: math.log(v) - math.log(first)
                          for h, v in day.items() if v and v > 0},
        })
    return pool


def weights(pool: list[dict], log_p: float, target_rolling: bool) -> list[float]:
    out = []
    for entry in pool:
        w = math.exp(-0.5 * ((entry["log_first"] - log_p) / SCALE_BANDWIDTH) ** 2)
        if entry["rolling"] == target_rolling:
            w *= ROLLING_BONUS
        out.append(w)
    return out


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Weighted quantile of (value, weight) pairs, linearly interpolated."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    acc = 0.0
    prev_v, prev_c = pairs[0][0], 0.0
    for value, w in pairs:
        acc += w
        c = acc / total
        if c >= tau:
            if c == prev_c:
                return value
            return prev_v + (value - prev_v) * (tau - prev_c) / (c - prev_c)
        prev_v, prev_c = value, c
    return pairs[-1][0]


def hour_stats(pool: list[dict], ws: list[float], hour: int):
    """(median, sigma, pairs) of the weighted log-ratios for one hour, or None."""
    pairs = [(entry["log_ratio"][hour], w)
             for entry, w in zip(pool, ws) if hour in entry["log_ratio"]]
    if not pairs:
        return None
    median = weighted_quantile(pairs, 0.5) * SHRINK
    q16 = weighted_quantile(pairs, 0.1587)
    q84 = weighted_quantile(pairs, 0.8413)
    sigma = max((q84 - q16) / 2.0, 0.05)
    return median, sigma, pairs


def smooth_series(values: list[float | None]) -> list[float | None]:
    """Replace each value by the mean of itself and its present neighbours.

    The comparable pool holds only a few dozen days, so one hour's median and
    spread are noisy estimates of a curve that really moves smoothly through
    the day; averaging over the adjacent hours trades a little bias for
    noticeably less variance. Missing hours (None) stay missing.
    """
    out: list[float | None] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        neighbours = [values[j] for j in (i - 1, i, i + 1)
                      if 0 <= j < len(values) and values[j] is not None]
        out.append(sum(neighbours) / len(neighbours))
    return out


def clean_quantiles(qs: list[float]) -> list[float | int]:
    """Clamp to the contest's range, enforce non-decreasing, round to 2 decimals."""
    out = []
    prev = 0.0
    for q in qs:
        q = min(max(q, 1.0), 1_000_000_000.0)
        q = max(q, prev)
        prev = q
        out.append(round2(q))
    return out


def forecast_case(case: dict) -> dict:
    known = case["known"]
    if isinstance(known, dict):
        known = [known]
    first = next((k for k in known if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p = max(1.0, float(first["viewers"]))
    levels = case.get("levels", LEVELS)
    pool = build_pool(case)
    target_rolling = is_rolling(case.get("notes", ""))
    ws = weights(pool, math.log(p), target_rolling)
    hours = [row["hour"] for row in case["forecast"]]
    stats = [hour_stats(pool, ws, h) for h in hours]
    medians = smooth_series([s[0] if s else None for s in stats])
    sigmas = smooth_series([s[1] if s else None for s in stats])
    rows = []
    for i, hour in enumerate(hours):
        if stats[i] is None:
            # No comparable speaks about this hour: persistence, widening spread.
            qs = [p * math.exp(_NORM.inv_cdf(t) * FALLBACK_SIGMA * math.sqrt(hour - 1))
                  for t in levels]
        else:
            median, sigma, pairs = medians[i], sigmas[i], stats[i][2]
            qs = []
            for tau in levels:
                empirical = weighted_quantile(pairs, tau) * SHRINK
                blended = median + GAMMA * 0.5 * (
                    (_NORM.inv_cdf(tau) * sigma) + (empirical - median))
                qs.append(p * math.exp(blended))
        rows.append({"hour": hour, "q": clean_quantiles(qs)})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
