"""GTA VI Twitch CCU forecaster (variant E): log-scale quantile forecast.

Model, in one paragraph: the hour-1 print anchors everything.  For every
past big launch bundled in the case file, take its release-day hourly curve
and write each later hour as a log ratio to that launch's own hour 1.  A
launch whose hour 1 is already huge stays near its hour-1 level all day; a
launch whose hour 1 is tiny explodes as the world wakes up and unlocks.
So, per forecast hour, the median log ratio is fitted as a saturating
("hinge") line in log hour-1 size: it falls steeply while hour 1 is small
and then flattens once hour 1 is large, which is exactly a morning decay
plus an evening bump for an ordinary launch (ratios dip to ~0.8x around
hour 6, climb through the afternoon, and peak near 2x at hours 19-20,
prime time, before easing off) and near-flat for a giant one.  The spread
is honest, not assumed: the five quantiles are the fitted median shifted
by the empirical quantiles of that hour's fit residuals across the
launches, so the band widens where launches genuinely disagree and stays
narrow where they agree.  Everything is fitted at run time from the
comparables inside the case file itself; no network, no stored model
weights beyond a small fallback curve used only if a case arrived with
too few usable comparables.

Run: `python3 -P -s /bot/main.py`, Python 3.13 standard library only.
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes one .out
per case to $MAKEFUN_OUTPUT_DIR (else /output).  Deterministic: the
same input bytes always produce the same output bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Fallback median log ratio to hour 1, hours 2..24, pooled from past
# launches: the morning dip, the climb, and the evening prime-time bump.
# Used only when a case carries too few usable comparables to fit.
FALLBACK_MED = [
    -0.138, -0.283, -0.310, -0.292, -0.238, -0.220, -0.107, -0.126,
    -0.092, -0.075, -0.022, 0.057, 0.166, 0.418, 0.502, 0.571,
    0.668, 0.876, 0.864, 0.846, 0.832, 0.780, 0.712,
]
# Fallback log spread (one sigma), growing slowly with the hour.
FALLBACK_SIG = [0.30 + 0.55 * math.sqrt((h - 1) / 23.0) for h in range(2, 25)]

MIN_COMPARABLES = 8      # fewer usable launches than this -> fallback
MIN_HOURS_PRESENT = 20   # of a launch's 24 release-day hours, to count it
MIN_SIDE = 5             # launches on each side of a hinge threshold


def parse_ts(text):
    """ISO-8601 timestamp (with a trailing Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_stamp(day_start, hour):
    """The 'from' stamp of release-day hour `hour` (1-based)."""
    moment = day_start.timestamp() + (hour - 1) * 3600.0
    return datetime.fromtimestamp(moment, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def release_day_curve(comp):
    """A comparable's 24 release-day hourly viewer counts, or None.

    Returns a list of 24 numbers (hours 1..24), with None where the
    source has no point for that hour.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_from = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
        except (KeyError, TypeError):
            continue
        if isinstance(viewers, (int, float)) and viewers > 0:
            by_from[row.get("from")] = float(viewers)
    return [by_from.get(hour_stamp(day_start, h)) for h in range(1, 25)]


def ols(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def hinge_fit(xs, ys):
    """Fit y = a + b*min(x, T) by least squares, T grid-searched.

    The hinge says the log ratio keeps falling as log hour-1 grows only
    up to a threshold launch size, then stays flat: small launches grow
    into the day, big launches are already there.  Returns (a, b, T).
    Falls back to a plain line when there are too few points to hinge.
    """
    n = len(xs)
    if n < 2 * MIN_SIDE + 2:
        a, b = ols(xs, ys)
        return a, b, float("inf")
    best = None
    for threshold in sorted(set(xs)):
        left = sum(1 for x in xs if x <= threshold)
        right = n - left
        if left < MIN_SIDE or right < MIN_SIDE:
            continue
        us = [min(x, threshold) for x in xs]
        a, b = ols(us, ys)
        sse = sum((y - (a + b * u)) ** 2 for u, y in zip(us, ys))
        if best is None or sse < best[3]:
            best = (a, b, threshold, sse)
    if best is None:
        a, b = ols(xs, ys)
        return a, b, float("inf")
    return best[0], best[1], best[2]


def quantile(sorted_values, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def normal_quantile(tau):
    """Standard normal quantile (Acklam's rational approximation)."""
    if tau <= 0.0:
        return -8.0
    if tau >= 1.0:
        return 8.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if tau < plow:
        q = math.sqrt(-2.0 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if tau > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def usable_curves(case):
    """Release-day curves of the case's comparables, usable for fitting."""
    curves = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve is None:
            continue
        first = curve[0]
        if first is None or first <= 0:
            continue
        present = sum(1 for v in curve if v is not None and v > 0)
        if present < MIN_HOURS_PRESENT:
            continue
        curves.append(curve)
    return curves


def hour_fit(curves, p1, hour, levels):
    """Fit one forecast hour: returns (median log ratio, per-level offsets).

    The offsets are the empirical quantiles of that hour's fit residuals
    across the launches (or normal-quantile times spread on the fallback
    path), so each quantile is median + offset in log space.
    """
    log_p1 = math.log(p1)
    if len(curves) < MIN_COMPARABLES:
        return FALLBACK_MED[hour - 2], [
            normal_quantile(tau) * FALLBACK_SIG[hour - 2] for tau in levels
        ]
    pairs = []
    for curve in curves:
        value = curve[hour - 1]
        if value is None or value <= 0:
            continue
        pairs.append((math.log(curve[0]), math.log(value / curve[0])))
    if len(pairs) < MIN_COMPARABLES:
        return FALLBACK_MED[hour - 2], [
            normal_quantile(tau) * FALLBACK_SIG[hour - 2] for tau in levels
        ]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    a, b, threshold = hinge_fit(xs, ys)
    # Never extrapolate the fitted line past the launches seen: clamp
    # the case's log hour-1 into the fitted range, then the hinge itself
    # flattens it above the threshold.
    x = min(max(log_p1, min(xs)), max(xs))
    med = a + b * min(x, threshold)
    residuals = sorted(y - (a + b * min(xx, threshold)) for xx, y in pairs)
    return med, [quantile(residuals, tau) for tau in levels]


def hour_quantiles(p1, med, offsets):
    """Turn one hour's fitted (median, offsets) into five quantiles."""
    raw = [p1 * math.exp(med + off) for off in offsets]
    # Keep every quantile a finite viewer count in (0, 1e9], non-decreasing.
    out = []
    floor = 1.0
    for q in raw:
        if not math.isfinite(q) or q <= 0:
            q = floor
        q = min(q, 1e9)
        q = max(q, floor)
        out.append(q)
        floor = q
    return out


def forecast_case(case):
    """Forecast hours 2..24 of one case, anchored on its hour-1 print."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    levels = case.get("levels") or LEVELS
    if len(levels) != 5:
        levels = LEVELS
    curves = usable_curves(case)
    rows = [int(row["hour"]) for row in case.get("forecast") or []]
    # Fit every hour independently, then smooth the fitted medians with
    # a 3-point moving average: neighbouring hours of one launch day
    # move together, so a jagged median is fitting noise, not signal.
    # (The per-hour spreads are left alone: they are already honest.)
    fits = {h: hour_fit(curves, p1, h, levels) for h in rows}
    medians = {}
    for i, h in enumerate(rows):
        neighbours = [fits[rows[j]][0] for j in (i - 1, i, i + 1)
                      if 0 <= j < len(rows)]
        medians[h] = sum(neighbours) / len(neighbours)
    hours = [{"hour": h,
              "q": hour_quantiles(p1, medians[h], fits[h][1])}
             for h in rows]
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
