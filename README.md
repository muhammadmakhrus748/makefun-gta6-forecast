[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/muhammadmakhrus748/makefun-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/muhammadmakhrus748/makefun-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecaster — variant E

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (hours 2–24, UTC), as five quantiles an
hour, from the first hour's print alone.

## The model

A log-scale quantile forecaster, anchored on the hour-1 print. For every
past big launch bundled in the case file, the release-day hourly curve is
rewritten as log ratios to that launch's own hour 1. The key regularity
in those curves: a launch whose hour 1 is already huge stays near its
hour-1 level all day, while a launch whose hour 1 is tiny explodes as
regions unlock and viewers arrive. So, per forecast hour, the median log
ratio is fitted as a saturating ("hinge") line in log hour-1 size — it
falls steeply while hour 1 is small, then flattens once hour 1 is large.
For an ordinary launch that shape is a morning decay plus an evening
bump: ratios dip to about 0.8x around hour 6, climb through the
afternoon, and peak near 2x at hours 19–20 (prime time) before easing
off; for a giant launch it is nearly flat. The spread is honest, not
assumed: each quantile is the fitted median shifted by the empirical
quantile of that hour's fit residuals across the launches, so the band
widens where launches genuinely disagree.

Everything is fitted at run time from the comparables inside the case
file itself. No network, no stored weights beyond a small fallback curve
used only if a case arrived with too few usable comparables.

## Run

`python3 -P -s /bot/main.py` (Python 3.13, standard library only).
Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`), writes one
`.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Deterministic:
the same input bytes always produce the same output bytes.

## Local validation

Leave-one-out over the 37 complete comparable launches in the kit's
practice cases: mean pinball loss **0.399**, against **1.142** for the
flat (all quantiles = hour-1 print) baseline. On launches with hour-1
prints of 50,000 or more — the regime a GTA VI-sized launch lives in —
**0.221** against **0.393** flat. 90% coverage 0.83, typical miss 78.8%
overall on the same leave-one-out run.
