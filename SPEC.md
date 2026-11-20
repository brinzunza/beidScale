# beidScale SPEC (pre-registered before any VM run)

Question: does large-scale learning on the BEID framework (raw-bar sequence models, millions of zones, a rich action grid, definition-level search, dynamic exits) produce a strategy with a net edge, and does scale change the answer?
Prior (stated up front): goal312 / goal315 / goal336 and the raw BEID baseline (goal335: about 0 R per trade, break-even spread under 1 pip) found no net edge; the most likely outcome is a documented null with flat scaling curves.

## Data
M1 mid OHLC 2005-2026 for EURUSD GBPUSD USDJPY EURGBP (OANDA S5 archive cache); 46-factor day vector from similaritySearch (known at the previous FX close). Zones: parameterized BEID machine (zones.py; defaults reproduce goal335 exactly) with daily reset (17:00 NY or 00:00 UTC) on M1/M5/M15/H1 bars; event = close of the confirming TF bar; simulation starts at the next M1 bar.
Zero-spread (gross) results are the headline; net results at a round-trip spread sweep 0 / 0.5 / 1.0 / 1.3 / 1.6 pips (net R = gross R - spread / risk_pips for filled trades).

## Action grid (Table A rewards, computed once by simulation)
420 actions = 6 entries (limit at 0/25/50/75/100% depth from the near edge; market at the event) x 5 stops (far edge, far edge - 0.25H, far edge - 0.5H, entry - 1 ATR15, entry - 2 ATR15) x 7 targets (0.5R, 1R, 1.5R, 2R, 3R, zone top, 2H) x 2 directions (with the zone; fade at the same price). Sell zones mirrored. Orders cancel at the order end (end of FX day); filled trades close at market after 48 h. Limit fill = first M1 bar touching the level (fill price = level); the fill bar can only lose; a bar reaching both stop and target = stop; stop-order fills for fades at exactly the level (optimistic, flagged). Actions with risk < 1.5 pips are invalid. Kernel verified against an independent per-bar reference (tests/test_rewards.py).
Known simplifications: no slippage, no swap/financing, entry exactly at level; one position per (pair, side, FX day) in evaluation (approximate de-overlap, earliest chosen event kept).

## Approach 1: policy learning over the table (dataset + epochs)
BEIDNet (conv/patch stems over 256 M1 and 96 M15 bars, context token with zone geometry + day vector + pair/tf/dow/type embeddings, transformer encoder, bf16, DDP): Q head (regression of net R at the training spread, all valid actions) + policy head (softmax over valid actions + skip, exact expected net R or log growth, entropy bonus). Sizes tiny 0.45M / small 6.7M / base 39M / large 115M.
Walk-forward, exit-purged, expanding; training years < Y-1, validation = Y-1 (epoch selection by validation Q loss), test = Y. Hyperparameters and decision rules chosen on 2012-2019 only; sealed 2023-2026.09 run once with frozen settings (pseudo-forward: these years were seen by earlier studies).
Decision rules fixed in advance: greedy policy (trade iff argmax is a trade action); greedyQ (argmax predicted net > 0); top-q (q = 1%, 3%) by policy margin over skip with the PREVIOUS year's margin quantile as threshold.

## Approach 2: strategy-definition embeddings
Table B: sampled zone-engine definitions (timeframe, reset, pullback-start rule, close vs wick confirmation, flip rule, minimum pullback bars, size bounds vs ATR15, minimum zone gap, session window) run over all pairs; per (definition, year, action) sufficient statistics (n, sum R, sum 1/risk, sum R^2; net at any spread = mean R - spread x mean 1/risk). Surrogate: MLP over (definition parameters, action attribute embeddings) -> mean R and mean 1/risk, 3-seed ensemble. Evaluation: 20% of definitions held out; for test year Y train on years < Y with the other definitions, select top-K (definition, action) among held-out definitions by LCB / mean / in-sample train performance / random, and report pooled realized net R in year Y (n-weighted, naive se; cells overlap heavily so se is optimistic) and the all-cell rank correlation. Then propose unseen definitions and verify them with real backtests (search.py).

## Approach 3: dynamic trade management (RL proper)
Episodes = filled trades of a base entry (default 50% limit, far-edge stop, 1R target); actions every K minutes: hold / exit / move stop to breakeven / trail 0.75R; conservative intrabar order (stop first); reward = realized net R at termination at 1.0 pip. PPO (own implementation) with GPU-vectorized replay (K = 1 reproduces Table A exactly, tested). Compared with hold-only (static Table A exit) and a random policy on the same test episodes.

## Controls and baselines
(1) Circular-shift nulls: rewards of events 5,000-60,000 rows away within the same pair, 8 shifts through the full pipeline; the real run must exceed the null maximum. (2) Planted edge: +0.10 R on one hidden action in a hidden state region (15m zones created 07-11 NY with height > 1 ATR15); the policy must choose it in >= 30% of region events. (3) Baselines on the same rows: best fixed action chosen on training years (1.0 pip), random valid action, action-conditioned GPU xgboost Q-model. (4) Scaling: model size {tiny, small, base, large} x data fraction {10, 30, 100}% on test years 2016/2019/2022: validation loss, expected net, top-q realized net R.

## Metrics
Per rule and spread: filled trades, fill rate, mean gross R, mean net R, month-clustered t (cluster = event month), win rate, trades per year, positive-year share, Kelly K = trades per year x mean^2 / var (0 when mean <= 0), median risk pips, share of wide stops / fades / >= 2R exits.

## Gate (decided before results)
A policy counts as a strategy only if, on the sealed block and on 2013-2022: net R > 0 at 1.3 pips with month-clustered t > 2; above the null maximum; K >= 1; >= 70% positive years; the planted edge is recovered; it beats the best fixed action and the xgboost baseline; and results improve (not just fluctuate) with scale. Only then a quant_engine.run_portfolio Monte Carlo ($50 per account, $1M target). Otherwise the result is reported as a null with the scaling curves.

## Multiplicity
Table A configurations: model sizes 4 x data fractions 3 x objectives 2 x rules 4; Approach 2: corpus definitions x 420 actions (all reported, selection evaluated out of time); Approach 3: base entries tried (all reported). Nothing is dropped.

## Amendment 1 (before any VM run): audit changes
Defaults and the gate are unchanged. Added as options, all reported if used: `--min_year` (drop 2005-07: coarse M1 feed and no day factors), `--tf_pow` (timeframe-balanced sampling), `--ema` (EMA weights for epoch selection and prediction), `BEID_NO_FADE=1` (ablation without fade actions), seed ensembles (`ensemble.py`), and a development-years-only hyperparameter search (`sweep.py`). Table A rows now also carry a `day_missing` flag and the 152 event features; the 46 day factors are mirrored to the long perspective for sell zones. PPO training episodes whose 48 h path reaches into the test year are purged.

## Amendment 2 (before any VM run): remaining audit fixes
`--select ev` (default): the epoch is chosen by the policy's expected net R on the validation year instead of the Q loss (`--select q` restores the old rule); `--include_val` + `refit.py`: refit on train + validation year with the winning epoch count; `--no_atr_pips`: remove the pips-scale feature that reveals the data era; `--fill_pen` / `BEID_FILL_PEN`: limit orders fill only when price trades through the level by that many pips (default 0 = touch, as before; 0.3 is the stricter setting); `BEID_BLOCK_HOURS=48`: no overlapping holds per (pair, side) in evaluation (default keeps one per FX day); the spread sweep gains 2.0 pips as an era stress. Defaults for the gate runs are otherwise unchanged; every variant that is run is reported.
