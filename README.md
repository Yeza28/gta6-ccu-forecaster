# gta6-ccu-forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Yeza28/gta6-ccu-forecaster/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Yeza28/gta6-ccu-forecaster/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch viewership forecaster. Given everything recorded up to the end of a game's release-day first hour (UTC) — that hour's concurrent-viewer count in the game's Twitch category included — it forecasts the five quantiles (5%, 25%, 50%, 75%, 95%) of every remaining hour of the day, hours 2 to 24.

The model is learned from the case itself: each case ships the hourly launch curves of dozens of comparable past game launches. Every comparable's release day becomes a curve of log-ratios to its own first hour. For each forecast hour the entry takes weighted quantiles of those log-ratios, weighting a comparable by how close its first-hour audience is to this case's (a Gaussian kernel on log scale) and by whether its unlock pattern matches (rolling midnight-local unlocks, read from the case text). The hour's quantiles blend the empirical weighted quantile with a normal fit around the weighted median, shrink the log-ratios 10% toward flat persistence, and anchor everything on the first hour's actual count. The hyperparameters (kernel bandwidth 0.65, rolling bonus 1.5, shrink 0.9, spread 0.95) were tuned by leave-one-out backtests over the comparable launches: on launches whose first hour already held a real share of their day (the regime a rolling-midnight console launch is in), the model scores a mean pinball loss of about 0.20 against about 0.27 for flat persistence.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `*.in` case file in the input directory and writes one `<id>.out` forecast per case to the output directory. Python 3.13, standard library only, no network, deterministic — the same input bytes give the same output bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=50nh4173&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
