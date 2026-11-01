# Event detection, control models, tracking features and tabular upgrades for possession from tracking

Research date: 2026-11-01. Goes beyond `research_notes/Posesión de balón en tracking/posesion_desde_tracking.md` and `balon_vivo_muerto.md`, and does not repeat them except where a number changes the decision.

Reading frame, from the repo: on leave-one-match-out IDSSE, the current model scores spellF1 0.480; the dead-ball oracle gives 0.516, the team oracle 0.909, no degradation at all 0.614. It predicts about 280-300 possessions against 437 real ones, and 30% of the real ones last under 2 s. Touch signals gave 0.488, touch plus change detector 0.496 (5 matches up, 2 down) — [docs/plan/continuar_m4_gpu.md](docs/plan/continuar_m4_gpu.md). So team-change evidence is the bottleneck, and dead ball is not.

Source quality notes: PLOS and arXiv details were read through full-text fetches summarised by a small model, so table values should be rechecked before they are quoted in the final document. "Measured here" means I ran it on this machine on 2026-11-01. Local code reading is cited by repo path.

## Rule-based event detection from tracking: thresholds, accuracy, data, licence, code

### Takeaway
The most honest published benchmark is Bischofberger et al. (2024): rule-based pass detection reaches F 0.87-0.92 when positional and event data are dependent, but only 0.71 on independent data, and the best thresholds change by a factor of up to 6.7 between datasets. Copying thresholds will not work; they must be fitted on degraded IDSSE. No public library does team-change detection from tracking alone without events or provider flags.

### Cited Findings
**Bischofberger, Baca, Schikuta (PLOS ONE 2024)** — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025972/), [eprints](https://eprints.cs.univie.ac.at/8072)
- Algorithm: a "hit" is ball acceleration above `min_acc` with the closest player within `min_vicinity`. A play is two subsequent hits by different players, or a hit then an interruption. Deflections are dropped when the previous play lasted less than `max_deflection_time` — [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11025972/fullTextXML)
- Fitted parameters (Table 4) and full-pipeline pass results (Table 6) — [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11025972/fullTextXML):

| Dataset | Source of positions | Games | min_acc (m/s²) | min_vicinity (m) | max_deflection (ms) | Pass P | Pass R | Pass F |
|---|---|---|---|---|---|---|---|---|
| Stats | Stats Perform, 10 Hz | 14 | 25.9 | 3.0 | 748 | 0.84 | 0.91 | 0.87 |
| Euro | Tracab, 25 Hz | 4 | 63.3 | 1.5 | 99 | 0.63 | 0.82 | 0.71 |
| Metrica sample | single video, 25 Hz | 3 | 26.8 | 1.4 | 91 | 0.89 | 0.90 | 0.89 |
| Subsequent | single video, 25 Hz | 6 | 9.5 | 3.0 | 487 | 0.88 | 0.96 | 0.92 |

- Matching tolerance was 500 ms (1000 ms for Euro) and the same player was required. Shot F was 0.58-0.77 — [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11025972/fullTextXML). The abstract's shot figure of 0.65 for independent data conflicts with 0.62 in Table 6 and in the Discussion.
- The authors say the results on Subsequent, Stats and Metrica are "likely an overestimation": those providers probably used event data to post-process positions, and in them the ball sticks to the possessing player. Only Euro is independent — [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11025972/fullTextXML)
- A small vicinity reduces false hits when a pass flies past a player, yet it must cover reach and noise; the authors say the algorithm cannot satisfy both. They also cite crowding around the ball and missing ball height. They doubt tackles and dribbles would score as well — [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11025972/fullTextXML)
- Code: no repository is given in the part read. The author's public repos are `accessible-space` (MIT), plus forks of databallpy and ETSY, with no event detector visible — [GitHub API](https://api.github.com/users/jonas-bischofberger/repos)

**Vidal-Codina et al. (2022)** — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804), CC BY per [MIT DSpace search result](https://dspace.mit.edu/handle/1721.1/145347)
- Possession zone 0.5 m for Track160, 1 m for Tracab and Hawk-Eye. Duel zone 1 m for all, with a duel flagged when at least two opponents are inside it, and "the duel condition prevails". Noisy ball is smoothed with Savitzky-Golay, order 2, 7 frames — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Passes exceed 90% precision and recall; shots are 78% / 53%; 70% of 254 shot errors were blocked shots labelled as passes; 569 crosses were labelled as passes — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- This 90%+ is the kind of figure Bischofberger's paper says is probably overestimated — [Bischofberger](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025972/). Bischofberger's overestimation claim concerns the field generally; I did not see it name Vidal-Codina.
- Its ball-change hyperparameters are only in Online Resource 1 and were not retrieved — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804). The earlier notes give databallpy's defaults instead (1.5 m, 5 m/s, 10°) — [databallpy docs](https://databallpy.readthedocs.io/en/main/features/team_and_player_possession.html)

**databallpy (MIT, 106 stars, pushed 2026-10-29)** — [GitHub API](https://api.github.com/repos/Alek050/databallpy), code inspected in [player_possession.py](https://raw.githubusercontent.com/Alek050/databallpy/main/databallpy/features/player_possession.py)
- A gain is valid when `min_frames_condition AND (angle OR speed)`. Speed is `ball_vel.diff().abs() > bv_threshold`, i.e. a one-frame difference. The angle uses one-frame vectors on entry and exit of the zone — [source](https://raw.githubusercontent.com/Alek050/databallpy/main/databallpy/features/player_possession.py)
- The file read contains no duel branch: `get_initial_possessions` just takes the closest player under the radius — [source](https://raw.githubusercontent.com/Alek050/databallpy/main/databallpy/features/player_possession.py). This nuances the earlier note saying rivals in the zone are treated as a duel.

**ETSY (KU Leuven; licence field NOASSERTION, 49 stars, last push 2024)** — [blog](https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization), [GitHub API](https://api.github.com/repos/ML-KULeuven/ETSY)
- Physical gates by event: pass-like max 2.5 m, foot height ≤1.5 m, ball must accelerate; interception or save max 2 m, ball must decelerate; bad touch 3 m; foul or tackle 3 m. Needs SPADL events. Coverage 91.61% on 313 matches, measured without ground truth — [blog](https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization)

**ELASTIC (KAIST)** — [arXiv 2508.09238](https://arxiv.org/html/2508.09238v1)
- Candidates: within 3 m of the ball, height under 3.5 m, local extrema of distance or peaks of smoothed acceleration. Scores use clipped linear terms: player-ball distance cap 3 m, ball acceleration cap 20 m/s², and "kick distance" (max player-ball distance between candidates) cap 5 m — [arXiv](https://arxiv.org/html/2508.09238v1)
- Receive frame on 1,706 events: 92.6% exact, 96.4% within 25 frames. All minor events (tackle, foul, bad touch, take-on, dispossessed): 44.4% exact, 77.6% within 25 frames — [arXiv](https://arxiv.org/html/2508.09238v1). It relies on the event sequence, so these are upper bounds for us.

**Other libraries**
- socceraction: MIT, 823 stars; un-xPass: licence NOASSERTION, 64 stars, last push 2024; floodlight: MIT; kloppy: BSD-3-Clause — [socceraction](https://api.github.com/repos/ML-KULeuven/socceraction), [un-xPass](https://api.github.com/repos/ML-KULeuven/un-xPass), [floodlight](https://api.github.com/repos/floodlight-sports/floodlight), [kloppy](https://api.github.com/repos/PySport/kloppy). floodlight and kloppy read provider flags rather than infer possession, per the [earlier notes](research_notes/Posesión%20de%20balón%20en%20tracking/posesion_desde_tracking.md).
- "Footballdata": the only packages found are [footballdataorg](https://github.com/hhllcks/footballdataorg), an API wrapper under GPL-3.0, and skagr's footballdata, incorporated in [soccerdata](https://soccerdata.readthedocs.io/en/latest/license.html). Neither appears to process tracking.

### Inferences
- The one-frame tests in databallpy fire on noise when applied to a noisy ball. Port the idea, not the code: measure the change over a filtered window on either side of the contact.
- Our own run should treat `min_acc`, `min_vicinity` and `max_deflection_time` as fitted parameters. Bischofberger's search bounds (0-120 m/s², 0.01-10 m, 0-1000 ms) are a ready grid. Optimise on degraded IDSSE by LOMO, never on the public 5 minutes.
- IDSSE is Tracab-type, so the Euro row is the realistic reference, and it warns of many false hits (precision 0.63). The Metrica sample row must not be used as an expectation; its ball sticks to the player, and the challenge tracking is uncorrected automatic data.
- "Kick distance" is the most useful borrowed idea. It needs ball-player distance growing around a contact, not a second derivative, so it should tolerate noise better. This is untested on our data.

### Gaps
- Vidal-Codina's ε values (Online Resource 1) were not retrieved.
- ETSY and un-xPass licence terms are not standard SPDX; read the LICENSE file before reusing code.
- I did not inspect socceraction or un-xPass code. I expect both to depend on event data but have not verified that here.
- No public benchmark tests rule detection on ball missing for 41% of frames.

## Pass and duel detection as possession evidence, including the challenge's proxies

### Takeaway
I found no published evaluation of "keeps the ball a few seconds" or "more than two consecutive passes" as a test of controlled possession. The best usable design is a two-sided retention test on each candidate change, but the local numbers cap what ball-based evidence can give: all ball restored gives only +0.080 spellF1.

### Cited Findings
- The challenge lists both proxies as possible signals, and excludes a touch without control, a deflection that does not intercept, and an aerial duel won without later control — [CHALLENGE.md](CHALLENGE.md)
- databallpy changes team only on a successful on-ball event by the opponent, which is event-derived — [docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
- Bischofberger's chained hits by different players and deflection window are the nearest tested primitive — [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11025972/fullTextXML)
- Vidal-Codina's rule that duels prevail over possession gives a tested way to freeze a change during contests — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Minor contact events are hard even with event data: 44.4% exact — [ELASTIC](https://arxiv.org/html/2508.09238v1)
- Local dose experiment: `ball_all` 0.568 against `ref41` 0.488; gaps placed on the changes give 0.467; real challenge gaps have a median of 10.7 s — [docs/plan/continuar_m4_gpu.md](docs/plan/continuar_m4_gpu.md)
- Mean possession is 7.2 s in IDSSE and 14.2 s in the public 5 minutes — [docs/plan/continuar_m4_gpu.md](docs/plan/continuar_m4_gpu.md)

### Inferences
- A 10.7 s median gap can swallow a whole chain of passes, so a chain feature must return "unknown", not "no chain", across gaps. Feed counts and an observed-share feature, with NaN where nothing was seen.
- Proposed features per frame, computed from detected hits with both past and future context (the task is offline):
  - `hits_team_A/B` in ±2, ±5 and ±10 s, and the distinct-player count for each team, with fragmented tracks handled by spatial continuity.
  - `first_team_with_two_hits_after_t`, and the time to it.
  - `retention_after_candidate`: seconds the new team stays nearest to the ball after a candidate gain.
  - `opponent_in_duel_zone`, `contest_duration`, and `chain_broken`.
  - `pass_return_gap`: whether the ball returns to the former team within `max_deflection_time`, as the deflection test.
- The proxies pull in opposite directions. Requiring three passes removes false changes, yet 30% of IDSSE possessions last under 2 s. They help if challenge truth is control-like, as the 14.2 s figure hints, but the IDSSE score could punish them. So use them as model features rather than hard rules, and keep the organisers' open question in mind.
- Testing on the undegraded IDSSE first is informative: the clean model is only 0.614. If a chain detector cannot beat that with a clean ball, it will not help under degradation, so that one run can kill the idea cheaply.

### Gaps
- No source quantifies the proxies, duel-winner detection from tracking, or interception detection without events.
- No source measures spell-level gain from pass chains. Any expected gain here would be a guess, so none is given.

## Pitch control and space features as evidence of who has the ball

### Takeaway
No evidence found that they help. I found no paper reporting possession classification from pitch control or Voronoi features, and the models are partly circular because they take ball location or possession as an input. Keep this as a low-priority, one-arm test.

### Cited Findings
- Fernández & Bornn's richer model uses player velocities and ball possession as team dominance inputs — [Springer](https://link.springer.com/10.1007/s10994-018-5725-1)
- Pitch control is defined as the probability a player would possess the ball if moved to a location — [arXiv 2503.19809](https://arxiv.org/pdf/2503.19809)
- A KNN approximation paper says these models "cannot be valued quantitatively" for lack of datasets; it reports no possession accuracy and only compares them visually — [arXiv 2501.05870](https://arxiv.org/html/2501.05870v1)
- Voronoi ignores velocity, per [Wu & Swartz](https://www.sfu.ca/~tswartz/papers/pitch_control.pdf)
- A Master's thesis reports 6-10% improvement in predicting goal-preceding events with pitch control. That is a different task, and I only saw a search summary — [thesis](https://staff.science.uva.nl/a.visser/education/masterProjects/Thesis_Ramon_Dop_MsDS.pdf)
- databallpy offers Voronoi, pitch control and team influence functions — [docs](https://databallpy.readthedocs.io/en/v0.5.0/_sources/api_docs/features_api.rst)
- The ball-free route with published team-level results remains PathCRF and Ball Radar — [earlier notes](research_notes/Posesión%20de%20balón%20en%20tracking/posesion_desde_tracking.md)

### Inferences
- The signal such features would add, who is closer and pressing, is largely already in the 46 features (team shape and ball-player distances). Velocity is the only new input, and fragmented tracks without identities make it unreliable.
- If tested, use one cheap variant: at the last seen ball position, the share of each team's reach by arrival time, plus a pressure count with no individual identity. Add it as a single LOMO arm and keep it only under the existing gate (+0.03 and at least 5 of 7 matches).
- Do not compute a full-pitch surface per frame. The cost is unjustified with no evidence of gain.

### Gaps
- No quantified result on this question exists in what I found. Rule out only after a local test.

## Tracklet stitching, team-consistent smoothing and ball filters

### Takeaway
Filter settings are well documented (Savitzky-Golay, order 2, 7 frames), but I found no study measuring the effect of filtering on event detection accuracy. Appearance-based stitching (GTA) solves identity, which we mostly do not need, so simple team-level continuity is the better investment.

### Cited Findings
- Savitzky-Golay order 2, 7 frames is used by Vidal-Codina — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804) — and is databallpy's default for positions, velocity and acceleration. databallpy's ball example uses a 25-frame window with clips of 50 m/s and 20 m/s² — [docs](https://databallpy.readthedocs.io/en/latest/getting_started/preprocessing_options.html). A 2026 paper uses order 2 with 21 points — [arXiv 2607.05845](https://arxiv.org/pdf/2607.05845) (search summary only).
- ELASTIC says acceleration from noisy position creates false peaks and uses Savitzky-Golay on velocity and acceleration, without giving the window — [arXiv](https://arxiv.org/html/2508.09238v1)
- Bischofberger fitted `min_acc` of 9.5-63.3 m/s², and notes that acceleration as a second derivative partly explains this spread — [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11025972/fullTextXML)
- A SoccerNet 2023 ball step keeps the detection closest to a smoothed trajectory within a gate and interpolates gaps linearly; another entry links player tracks without appearance and imputes with Gaussian-process interpolation — [arXiv 2308.16651](https://arxiv.org/pdf/2308.16651) (search summary only)
- GTA (MIT licence) splits and joins tracklets by appearance. SORT rises from HOTA 65.89 to 72.73, with ID switches from 3281 to 1374; the paper's text also gives 79.41 to 83.11. I could not reconcile the two statements — [arXiv 2411.08216](https://arxiv.org/pdf/2411.08216), [licence](https://raw.githubusercontent.com/sjc042/gta-link/main/LICENSE)
- The Kalman results found concern video or robot ball tracking, not event accuracy — [survey](https://arxiv.org/pdf/2204.04143), and vary by dataset and metric.
- Vidal-Codina names player swaps in tracking as a cause of wrong events — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)

### Inferences
- A 25-frame window (1 s) is wider than most contacts, so it would flatten the very peaks a detector needs. Do not reuse databallpy's example on detection features, nor its 20 m/s² clip, which sits under all but one of Bischofberger's fitted thresholds.
- A sensible order is: reject outliers by a speed gate, split at gaps, then filter each contiguous segment. Never filter or differentiate across a gap, since interpolation makes artificial speed.
- A constant-velocity Kalman or RTS smoother assumes no sudden changes, and a kick is one. If tried, use change-aware segments: detect contacts, then fit each flight separately. Keep Savitzky-Golay as the baseline, and compare by hit precision against IDSSE truth on noised data.
- Our features are per-team sets, so identity matters only for "same player keeps the ball". Replace identity with a spatial rule: same team, within a speed-feasible distance of the prior holder. Team-label flips on a fragment could be corrected by a majority vote over the fragment. This needs checking on our tracks.
- The degradation in `train.py` adds 0.25 m noise smoothed over 25 frames to positions. Check whether the IDSSE ball carries realistic high-frequency noise. Otherwise, thresholds fitted there will not transfer to the challenge ball.

### Gaps
- No source gives the effect of ball smoothing or stitching on pass or possession accuracy. Everything above on that effect is engineering judgement to be tested.
- No source on team-consistent stitching in football was found.

## Tabular model upgrades: LightGBM, CatBoost, XGBoost, lag features, constraints, losses, weights, calibration, ensembling, determinism

### Takeaway
Switching library is a poor first bet and not required for determinism: all four libraries gave identical hashes across thread counts in my test. The more valuable changes are in how the current model is set up: leak-free early stopping, a split into dead and H/A heads, and multi-scale future and past lags.

### Cited Findings
**Measured here**, 2026-11-01: 150,000 synthetic rows, 46 features with 20% NaN, 3 classes, 120 rounds, 63 leaves where applicable. SHA-256 of `predict_proba` on 20,000 rows was compared across 16, 16, 4 and 1 threads, each in a fresh process. Windows 11, 16 threads. Scripts were in the session scratchpad and are not in the repo.

| Model | Version | Hash identical in all 4 runs? | Fit + predict time at 16 threads (s) |
|---|---|---|---|
| sklearn HGB | 1.6.1 | Yes | 36-46 |
| sklearn HGB, `max_features=0.7` | 1.6.1 | Yes | 25-33 |
| LightGBM, default | 4.7.0 | Yes | 17-19 |
| LightGBM, `deterministic` + `force_row_wise` | 4.7.0 | Yes | 25-27 |
| XGBoost, `hist`, lossguide | 3.4.1 | Yes | 23-36 |
| CatBoost, depth 6 | 1.2.10 | Yes | 7-8 |

Timings are not a fair comparison: other jobs were running in parallel, and CatBoost used a different tree shape. HGB at 4 threads took 11.6 s, far less than at 16, which suggests oversubscription under that load; it does not prove it.

**Documentation**
- LightGBM: `deterministic` should give stable results with the same data and parameters, even with different `num_threads`, and needs `force_row_wise` or `force_col_wise` set. Different versions, compilers or systems are expected to differ — [LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.4.0/Parameters.html)
- CatBoost: on CPU, thread count "doesn't affect results" — [CatBoost docs](https://catboost.ai/docs/en/references/training-parameters/performance.md)
- XGBoost: results "could" differ between runs because of floating-point summation order and multi-threading — [XGBoost FAQ](https://xgboost.readthedocs.io/en/stable/faq.html). My test did not show this, but one test does not refute it.
- sklearn: an integer `random_state` gives reproducible output. The page does not discuss OpenMP — [sklearn HGB](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- sklearn's `monotonic_cst` is valid only for binary classification; `log_loss` is the only documented loss; `class_weight` multiplies `sample_weight`; early stopping is on by default above 10,000 samples, with a `validation_fraction` of 0.1, and `X_val` is accepted from 1.7 — [sklearn HGB](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

**Comparative evidence**
- TabArena: with post-hoc ensembling, TabM first, LightGBM second, RealMLP third, CatBoost fourth (first without ensembling), XGBoost fifth — [arXiv 2506.16791](https://arxiv.org/pdf/2506.16791). Elo values change by leaderboard version.
- An older comparison found CatBoost best with small differences and LightGBM the fastest — [Bentéjac et al. summary](https://www.datalearner.com/academic/journal-papers/0269-2821/volumes-and-issues/363/paper-detail/3530)
- No source compares sklearn HGB directly with them on frame data.

**Losses**
- Class-balanced GBDT losses improved F1 on 13 of 15 binary sets (0.38% to 28.91%, two losses), and 14 of 15 multi-class sets (generally under 2%). Weighted cross-entropy gave the most consistent gains — [arXiv 2407.14381](https://arxiv.org/html/2407.14381v1)
- Focal loss on LightGBM binary beat cross-entropy on 8 of 15 sets and lost on 7, from a count of Table 1 made by the fetching model and not rechecked by me — [arXiv 2407.14381](https://arxiv.org/html/2407.14381v1). LightGBM needs custom gradients for it — [Halford](https://maxhalford.github.io/blog/lightgbm-focal-loss). Robust-GBDT's claims are from its own authors — [arXiv 2310.05067](https://arxiv.org/pdf/2310.05067)

**Sequence context**
- Lang et al. reach 92% with shifted copies at ±0.4 to ±80 s — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- PathCRF shows Viterbi with constraints moving event F1 from 65.34 to 73.14 — [arXiv](https://arxiv.org/html/2602.12080)

**Local code**
- `fit()` builds `HGB` without `early_stopping` and without validation data, and it stacks each frame with its home/away swap — [possession/train.py](possession/train.py). With sklearn's default, a random 10% split is then used.

### Inferences
- **Probable leak in early stopping.** Rows are every 5th frame and each has a swapped twin, so the random 10% holdout is nearly a copy of training, and stopping is ineffective. Print `n_iter_` to confirm. Then either set `early_stopping=False` with a fixed iteration count chosen by LOMO, or hold out a whole match. With sklearn 1.6.1 installed, `X_val` is not available without upgrading.
- **Split the head.** Train `P(dead)` and `P(H | live)` as two binary models. The latter is what the team oracle says matters, and it can use `monotonic_cst` in sklearn. Candidates are antisymmetric features such as "ball nearer to H" (+1). Enforce H/A symmetry by averaging the prediction on a frame and its swap, which is deterministic. Monotonic constraints on noisy ball evidence are an untested idea; no source tests them in this setting.
- **Weighting near transitions.** Weighted cross-entropy has the best general evidence. Weight the frames within ±1 s of a real H↔A change, and those in short possessions. Test 2 and 5 times the weight. Short possessions are what the model loses, but this also changes probabilities, so recalibrate after.
- **Avoid focal loss for now.** The evidence is mixed, it needs a custom objective, sklearn cannot do it, and it distorts the probabilities that feed the decoder.
- **Calibration.** The decoder sums log-probabilities, so calibration should shift its costs. Fit a single temperature per class on out-of-fold LOMO predictions, then re-select `switch` and `dead` costs. It is cheap, closed-form and deterministic. Isotonic is riskier here, since it has steps.
- **Lags.** Add a compact set at ±0.4, ±2, ±8 and ±20 s of the best few signals, and specifically the future side. Rolling mean, max and last-seen-age are more robust than raw shifts with missing ball.
- **Ensembling.** Average 3-5 seeds or subsample variants. With `max_features` in HGB, determinism held in my test. Only ensemble after the above: it improves probabilities but cannot create evidence that is missing.
- **Library swap as one arm.** Use LightGBM 4.7.0 with the same features, since it trains quickly and supports weights and custom objectives. CatBoost's ordered boosting and symmetric trees are a useful diversity source for an ensemble.
- **Byte-identical regeneration.** Pin exact versions and wheels, record `OMP_NUM_THREADS`, and test a second machine. Documentation warns only about version or platform changes, and my test does not cover those. Keep a CI-style check that hashes the final CSV.

### Gaps
- Cross-machine and cross-OS byte identity was not tested, and neither was a real-data run. LightGBM `bagging_fraction` or `feature_fraction` were not tested either.
- No evidence was found on calibration before Viterbi decoding, sample weighting near transitions, or monotonic constraints in sports tracking.
- XGBoost's `nthread` determinism has no official guarantee that I could find.

## Dead-ball improvements with Smart Tagging type signals

### Takeaway
The biggest overlooked item is not about dead ball: Smart Tagging has 602 team-tagged phases, including attacking and defensive transitions, and the pipeline uses it only for kick-off timing and a plausibility check. It is direct team-change evidence, with one catch: IDSSE has no such tags, so LOMO cannot measure it unless the tags are simulated. For dead ball, gains are capped at +0.036 by the oracle.

### Cited Findings
- 638 automatic phases and set pieces with start and end. Team tags: Barcelona 314, Athletic 288, N/A 36. Codes include `SET PIECES` 59, `ATTACKING TRANSITION` 24, `DEFENSIVE TRANSITION` 24, `BUILD UP` 22, `DEFENDING IN MIDDLE THIRD` 66, `PROGRESSION` 66. Set-piece types: throw-in 26, corner 12, goal kick 10, free kick 5, kick-offs 6 — [DATA.md](DATA.md)
- Organisers confirm that use is allowed. It has no manual review and holds false positives and negatives; "every phase carries a team tag and every set piece is a restart"; intervals "are phases, not possessions" — [FAQ.md](FAQ.md)
- `possession/pipeline.py` uses it for kick-off start frames and `set_piece_check`. Grep finds no Smart Tagging use in `features.py` or `train.py` — [possession/pipeline.py](possession/pipeline.py)
- Dead-ball oracle: 0.480 to 0.516 — [docs/plan/continuar_m4_gpu.md](docs/plan/continuar_m4_gpu.md)
- Ball gaps located in dead ball would be favourable: `dead41` +0.051, `dead_m4` +0.076 over `ref41` — [docs/plan/continuar_m4_gpu.md](docs/plan/continuar_m4_gpu.md)
- I found no public Metrica documentation on how Smart Tagging works — [search result](https://arxiv.org/pdf/2606.09289) returned only unrelated work; the closest is a 2026 paper on in-possession phase boundaries by temporal graph learning.
- Restart types are confused near corners and sidelines, per Vidal-Codina — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804). Lang's ±2 s boundary rates and 3.64 s median filter are in the [earlier notes](research_notes/Posesión%20de%20balón%20en%20tracking/balon_vivo_muerto.md).

### Inferences
- **Phase team as emission prior.** Convert each phase to per-frame log-odds for H or A, added to the model's emissions before Viterbi. `BUILD UP`, `PROGRESSION`, and `BALL IN FINAL THIRD` plausibly mark the tagged team in possession, and `DEFENDING *` probably marks the tagged team without it. The meaning of the team tag for each code must be established first, using only structure (for example where the ball is relative to the tagged team's goal), never by tuning on the public 5 minutes' labels. Whether inspecting them for meaning is acceptable is the question already raised in the plan.
- **Transitions as change locations.** Attacking and defensive transitions (24 each) mark moments where a change is likely near the start of the interval. Lower the switch cost in a window around them. This addresses only a small share of about 437 changes, but they are the changes our model cannot see.
- **Measuring without labels.** Simulate tags in IDSSE from its truth with matched false-positive, false-negative and offset rates, across a pessimistic range, then measure gain in LOMO. Use weak checks on the challenge match: agreement between tags and model outside the graded public window. Because tag quality is unknown, keep the prior weak and state this in the technical document.
- **Dead ball.** Keep it simple: use each `SET PIECES` window as a soft prior that a restart exists, and let the `Type` tag choose the restart spot check (sideline for throw-in, corner mark, goal area). Snap the dead-to-alive boundary to the first hit detected inside the window. Set-piece team tags may also give the team after the restart, which is direct H/A evidence right at a boundary.
- Do not make tags a boosted-tree feature. The model trains on IDSSE, which has none, so the tree would never learn them. Apply them at the decoder level.
- Reported order of work, from strongest expected value per effort:
  1. Check early-stopping leak and split the heads (a day, no new data).
  2. Smart Tagging team prior at the decoder, with simulated tags on IDSSE.
  3. Filtered hits, chains and retention features, first on clean IDSSE as a kill test.
  4. Calibration, weights near transitions, seed averaging.
  5. LightGBM arm, then pitch-control arm, only if time remains before 17 Nov.

  This order is my judgement, not a measured result; only the oracle figures behind it are measured.

### Gaps
- Precision and recall of Smart Tagging's team tags, boundaries and semantics are unknown; no public documentation was found and no check has been run.
- The share of real team changes near a tagged phase start is unknown, and it decides how much item 2 can give. It can be measured on the public 5 minutes for understanding, but not for tuning.
- No published dead-ball method uses signals of this kind.
