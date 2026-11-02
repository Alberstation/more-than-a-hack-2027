# Domain adaptation, label definition and label noise for the possession model

Research date: 2026-11-01. Items tagged **[measured]** were computed in this session from stored v3 predictions (`mixed_all3`, out-of-fold on IDSSE, trained on all 7 IDSSE matches for Metrica). No challenge data, public labels or challenge predictions were used. No model was retrained. Decoder is the repo's Viterbi with dead cost fixed at 40 and gain 0. Scores use `train.score`, which calls the official functions. "u" and "d" are the two rehearsal gap patterns (uniform, dead). Scripts are temporary, in the session scratchpad (`hedge/hedge.py`, `hedge/m2.py`).

## Which definitions exist, and can a post-hoc rule hedge raw against ctrl?

### Takeaway
**Yes, and this is the main new finding.** Raising only the H/A switch cost of the raw-trained v3 gives most of the ctrl benefit at almost no raw cost. At switch cost 14 to 20 (now 5), Metrica loses 0.000 to 0.005 against its event labels and gains +0.023 to +0.037 against a Metrica control-style label. It clears +0.03 under control only at 20. Merging short spells by duration and blending with a ctrl model are both no better, and duration merging is clearly worse.

### Cited Findings
**Definitions**
- Metrica: possession starts when a player touches the ball and is not lost on insignificant opponent interference — per [Metrica event definitions](https://github.com/metrica-sports/sample-data/blob/master/documentation/events-definitions.pdf), quoted in the earlier note.
- Challenge: a change needs an opponent gaining **controlled** possession; touches and undisturbing deflections do not count. Truth takes the team the tracking names as owning, unless the tracking marks the ball dead — [CHALLENGE.md](../../CHALLENGE.md).
- Link & Hoernig: individual possession begins when a player "is able to perform an action with the ball", without needing a touch. Team possession ends at the opponent's first individual possession. Control requires choosing between play options. Contacts in midfield "ping-pong" and headed-on balls are ball actions without control. No explicit duel rules are given. Team control intervals last about 1.2 s and total 17:49 per match, against 56:04 of team possession — [PLOS ONE 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225).
- Vidal-Codina: a possession zone around each player (0.5 m for provider A, 1 m for B and C) and a 1 m duel zone around the ball. Two opponents in the duel zone make it a duel, which takes precedence. A **gain requires a change in ball direction or speed**, otherwise the frames are dropped as a ball passing nearby. A loss requires the ball to leave the zone; regaining it untouched by anyone else records no loss. Thresholds are in the supplement. Dead ball comes from a flag (provider B) or missing ball data (A, C). The paper says no possession gold standard exists — [arXiv 2202.00804](https://arxiv.org/html/2202.00804v3).
- Stats Perform/Opta: a possession is ended by the opposition gaining control, and one possession may contain several sequences, for example a saved shot leading to a corner. One source says the Opta Pro method is not public. These come from search summaries because the page returned HTTP 403 — [Stats Perform](https://www.statsperform.com/resource/sequences-and-the-premier-league).
- StatsBomb: `possession_team` is the team that started the possession "in control of the ball" — [socceraction schema](https://socceraction.readthedocs.io/en/stable/api/generated/socceraction.data.statsbomb.StatsBombEventSchema.html). A related community method counts passes, dribbles, shots and fouls won, and excludes clearances — [American Soccer Analysis](https://www.americansocceranalysis.com/home/2018/9/4/expected-goal-chains-are-back).
- Shelopugin: possession is a series of control actions by one team uninterrupted by the opponent — [arXiv 2406.00814](https://arxiv.org/html/2406.00814v1). This is a later extension, not Fernández and Bornn's own definition, which was not confirmed.
- unravelsports infers ball-owning team from distance to the ball when absent and warns about contested and aerial balls — [docs](https://unravelsports.readthedocs.io/en/latest/api/soccer/dataset.html).

**[measured] Definition gap.** Using the other label as a perfect prediction gives spellF1 0.596 on IDSSE. On Metrica, the event label against a control-style label gives 0.747. So perfect raw knowledge still scores only about 0.6 if truth is ctrl on IDSSE.

**[measured] A control-style label on Metrica.** The same rule as `control_labels`, with PASS, SHOT and SET PIECE as evidence and the ±1.5 s window fixed earlier: Game 1 goes from 385 spells of 9.2 s to 250 of 14.1 s, and Game 2 from 371 of 9.0 s to 245 of 13.6 s. It relabels 4.3% of frames. IDSSE ctrl is 227 of 13.9 s, so the two providers converge.

**[measured] Switch cost frontier, paired against cost 5.** IDSSE has 7 matches (wins out of 7). Metrica has 6 degraded runs of 2 games.

| Switch cost | IDSSE vs raw (u / d) | IDSSE vs ctrl (u / d) | Metrica vs event label (u / d) | Metrica vs control-style (u / d) | Spells, IDSSE |
|---|---|---|---|---|---:|
| 7 | −0.004 / +0.001 | +0.018 / +0.023 (7/7) | +0.000 / +0.001 | +0.006 / +0.010 (6/6) | 310 |
| 10 | −0.010 / −0.006 | +0.030 / +0.036 (7/7) | −0.002 / −0.001 | +0.011 / +0.018 (6/6) | 295 |
| **14** | −0.015 / −0.008 | +0.042 / +0.058 (7/7) | −0.003 / −0.000 | +0.023 / +0.027 (6/6) | 282 |
| **20** | −0.023 / −0.020 | +0.062 / +0.074 (7/7) | −0.004 / −0.005 | +0.031 / +0.037 (6/6) | 266 |
| 28 | −0.038 / −0.029 | +0.075 / +0.093 (7/7) | −0.017 / −0.017 | +0.038 / +0.043 (6/6) | 251 |
| 40 | −0.050 / −0.051 | +0.087 / +0.106 (7/7) | −0.042 / −0.033 | +0.038 / +0.053 (6/6) | 236 |

Against raw, cost 14 won only 0/7 (u) and 1/7 (d) on IDSSE, so the small loss is consistent. Worst single-match raw loss at cost 14 was −0.023. Baseline here reproduces the known 0.556/0.596 and 0.602/0.643.

**[measured] Other hedges, as IDSSE mean of u and d, raw then ctrl.**

| Option | vs raw | vs ctrl | Verdict |
|---|---:|---:|---|
| Cost 5 (reference) | 0.576 | 0.615 | |
| Cost 20 | 0.555 | 0.682 | Frontier |
| Cost 28 | 0.543 | 0.698 | Frontier |
| Merge A-B-A under 2 s (on cost 10) | 0.552 | 0.671 | Dominated by cost 20 |
| Merge under 3 s | 0.515 | 0.685 | Dominated by cost 28 |
| Merge under 5 s | 0.462 | 0.671 | Much worse |
| Blend 0.5 raw + 0.5 ctrl log-probs, cost 10 | 0.505 | 0.717 | About on the frontier, so no hedge gain |

- On Metrica, merging under 3 s loses 0.04 against event labels, against 0.004 for cost 20.
- Leave-one-match-out choice of cost, by mean of raw and ctrl, picked 28 in 6 of 7 folds, giving 0.543 / 0.695. Max-min picked 5 in all folds, giving 0.576 / 0.615. That rule is pulled to the raw side because raw is always the lower score, so use regret instead.

**[measured] Earlier rejection, reread.** The stored summary shows raw-trained v3 with costs tuned for ctrl at 0.697 (u) and 0.725 (d), against 0.711 and 0.750 for `ctrl_mixed_all`. So training on ctrl adds only +0.014 to +0.025 beyond what decoder cost gives. The earlier Metrica penalty of about −0.05 for `costs_for=ctrl` used cost 40 and dead 80. Intermediate values were not tested at that time.

### Inferences
- **Why it works:** a higher switch cost removes short spells that have weak evidence, whereas a duration merge removes them regardless of evidence. This matches the control definition, and Vidal-Codina's requirement of physical evidence for a gain.
- Expected gain against the current setup, for a probability π that truth is ctrl-like:

| Cost | Break-even π, IDSSE | Break-even π, Metrica | Gain at π=0.3, IDSSE / Metrica | Gain at π=0.5, IDSSE / Metrica |
|---|---:|---:|---:|---:|
| 14 | 0.18 | 0.06 | +0.007 / +0.006 | +0.020 / +0.012 |
| 20 | 0.24 | 0.12 | +0.006 / +0.007 | +0.023 / +0.015 |
| 28 | 0.29 | 0.30 | +0.002 / +0.000 | +0.025 / +0.011 |

- **Recommendation:** freeze cost 20 as the likely best, or 14 if the team wants the smaller raw risk. My judgement is that π is above 0.25, based only on the rule text ("controlled", touches do not count) and Metrica's published definition. This is a judgement, not a measurement.
- **π must not come from the public 5 minutes.** The 17 spells of 14.2 s would be the obvious evidence, and using it would be the banned selection. Nor should the prediction count on the challenge be used. Stating π and cost in the log before looking at any new output keeps this clean.
- Validation is already mostly done, as IDSSE and Metrica agree in sign on all cells at cost 14 or above. To confirm: rerun in the production pipeline across all degradation draws, and record the frozen choice. This meets the byte-identical rule, since it changes one constant.

### Gaps
- Metrica supports only 2 independent games. Its 6 runs are not independent, and the control-style label is built from the same family of rule as the IDSSE one, so agreement partly reflects shared construction.
- Costs here used dead cost 40, which equals production v3 (`SWITCH_COST` 5, `DEAD_COST` 40 in [train_v3.py](../../possession/train_v3.py)). The interaction with dead cost was not swept.
- Differences at this size sit below the repo's 0.04 single-match noise band. Paired signs are strong, but the absolute expected gain on one match is uncertain.
- Real challenge truth might be neither label. Opta and StatsBomb definitions were read only through summaries, and the StatsBomb specification PDF could not be read.
- No public DFL/Sportec document defining `BallPossession` beyond the earlier note was found.

## Noisy and ambiguous labels: soft labels and boundary tolerance

### Takeaway
Boundary jitter is the only untested idea worth an arm. Its evidence is from video and imaging only. Global smoothing has no clear case, and may hurt at high noise.

### Cited Findings
- One second of lateness gives spellF1 0.654, and 8 frames gives 0.893 — [EVALUATION.md](../../EVALUATION.md).
- Temporal Label Smoothness rolls targets by a random step up to ε, so only frames near boundaries change. Its authors found losses near boundaries stay higher than elsewhere — [DIR-AS](https://arxiv.org/pdf/2304.02110).
- Adaptive boundary smoothing scales the mixing width with estimated action duration — [arXiv 2207.08653](https://arxiv.org/pdf/2207.08653).
- Label smoothing improves calibration but harms distillation — [Müller et al.](https://arxiv.org/abs/1906.02629v2).
- Its advantage vanishes at high label noise, where negative smoothing can do better; lower noise favours positive rates — [Wei et al., ICML 2022](https://arxiv.org/abs/2106.04149).
- Programmatic weak supervision can estimate labeler accuracy without truth, with identifiability under mild conditions — [Yu et al.](https://arxiv.org/pdf/2106.04530). The disagreement survey is [Uma et al.](https://iris.unibocconi.it/bitstream/11565/4051365/1/2021_jair.pdf).
- `two_heads` was already within ±0.01 — [BITACORA.md](../../BITACORA.md).

### Inferences
- **Arm to test:** raw labels with transitions shifted by a random offset per training draw, up to about 0.5 s. This fits the current boosted-tree setup without a new loss, because it only changes `y`. Soft targets could be imitated by duplicating boundary rows with weights, at the cost of file size.
- Validate by paired leave-one-match-out on IDSSE, with the same fixed setting scored on Metrica. Adopt only with a gain of at least 0.01 on both. Expected gain is unknown.
- Multi-labeler methods mainly solve the same problem as the frontier above, with more machinery. Skip them.

### Gaps
- No study of boundary smoothing with gradient boosting or sports tracking was found.
- No study of noise-robust losses with boosted trees was found.

## Unsupervised domain adaptation for tabular and sequence data

### Takeaway
No method has dependable evidence over plain gradient boosting, and with one unlabelled target there is no reliable way to tune one. Only per-match feature normalisation deserves a cheap trial.

### Cited Findings
- SKADA-Bench tabular columns (mean accuracy, Mushrooms / Phishing, source-only 0.72 / 0.91): CORAL 0.72 / 0.92, discriminative reweighting 0.75 on Mushrooms. Base estimators were selected among logistic regression, RBF-SVM and XGBoost. Only two tabular sets are included — [SKADA-Bench](https://ar5iv.labs.arxiv.org/html/2407.11676).
- On simulated data, reweighting gains largely disappear with a stronger base classifier. Gaussian reweighting collapsed on real data (Phishing 0.46) — [SKADA-Bench](https://ar5iv.labs.arxiv.org/html/2407.11676).
- CORAL lost on 3 of 12 columns, gained on 6 and tied on 3 — same source. Comparison is of raw means, not the paper's significance marks.
- Unsupervised scorers are weak. Correlation with target accuracy was 0.98 for a supervised scorer, 0.71 for circular validation, 0.56 for importance weighting, and at most 0.06 for SND, DEV and PE — same source.
- CORAL aligns second-order statistics with no target labels — [Sun et al.](https://arxiv.org/pdf/1511.05547).
- Prior test-time methods often fail to beat the unadapted model on tabular data, and unsupervised objectives can collapse. AdapTable adjusts outputs only, so it works with boosted trees, and raised CatBoost on several sets but dropped Voting balanced accuracy from 80.4 to 79.6 — [AdapTable](https://arxiv.org/html/2407.10784v4).
- Batch norm adaptation does not apply to trees, since they have no such layers (basic fact of the model class, no source needed beyond the [repo model](../../possession/train_v3.py)).
- `self_train` already gave nothing — [BITACORA.md](../../BITACORA.md).

### Inferences
- A feature-level alignment trial is cheap: z-score scale-dependent features per match, or apply CORAL from degraded IDSSE to the target's own covariance. Both are fully automatic and deterministic.
- Validate by fitting on IDSSE, aligning to each Metrica run with no labels, and scoring with fixed settings. Require a paired gain in both gap patterns. Prior probability of reaching +0.03 looks low.
- Do not do covariate weighting beyond the discriminator diagnostic proposed in the earlier note. One target match cannot select a stable setting.

### Gaps
- No football-tracking adaptation result was found. SKADA-Bench's XGBoost-specific results were not separated.
- Only the first 100,000 characters of AdapTable and SKADA-Bench were read, so limitations sections may be incomplete.

## Team-style invariance and prior shift

### Takeaway
**Closed by measurement:** even an oracle H/A share gives no spellF1 gain, so Saerens EM should be dropped. High possession teams are outside the tested range only slightly.

### Cited Findings
- Saerens EM needs no retraining and depends on calibration. The likelihood is concave and bounded, and bias-corrected calibration works better than temperature scaling — [Alexandari et al.](https://arxiv.org/abs/1901.06852v5).
- Label shift has been described as a main driver of tabular loss — via [AdapTable](https://arxiv.org/html/2407.10784v4), citing TableShift.

**[measured] Prior adjustment on v3, paired against cost 10, with weight w on the log-prior ratio, H and A only.**

| Prior used | w | IDSSE vs raw, u / d | IDSSE vs ctrl, u / d | Metrica, u / d |
|---|---:|---:|---:|---:|
| EM | 0.25 | −0.000 / −0.001 | +0.003 / +0.002 | +0.001 / −0.001 |
| EM | 1.0 | −0.009 / −0.013 | −0.004 / −0.006 | −0.002 / −0.006 |
| **True share (oracle)** | 0.25 | −0.002 / −0.000 | +0.003 / +0.001 | −0.001 / +0.001 |
| **True share (oracle)** | 1.0 | −0.004 / −0.007 | +0.002 / +0.000 | +0.000 / −0.006 |

- EM does recover the share reasonably: for J03WPY it gives 38.5/16.7 against a true 35.0/20.7, overshooting, and J03WR9 gives 37.3/19.8 against 36.3/21.8.
- The IDSSE range tested reaches about 63% of alive frames for one team.

### Inferences
- The frame features already reveal which team has the ball, so the frame prior carries little extra information. Note that a per-frame bias accumulates across a spell in Viterbi: a bias of 0.1 over 100 frames equals a switch cost of 10. This is why strong w damages spells.
- For Barcelona at perhaps 65–70%, this is mild extrapolation. A prior correction has no measured case. Do not include it.
- Mirror augmentation is already covered: `symmetric` was within ±0.01 — [BITACORA.md](../../BITACORA.md). A sports video flip with team swap exists as practice — [int8.io](https://int8.io/team-ball-action-spotting-challenge-2025), but mirror invariance in vision is not automatic — [liner summary of Visual Chirality](https://liner.com/review/visual-chirality).
- Tempo normalisation has no evidence for or against. Barcelona's slower spells are absorbed by the higher switch cost, in part.

### Gaps
- No published work on team-style invariance for possession models was found.
- Tempo normalisation was not measured. IDSSE contains no team as extreme as the target may be.

## Better simulation of degradation

### Takeaway
No learned noise model exists for football. One cheap addition has transferable evidence: temporally correlated Ornstein-Uhlenbeck position noise, with severity randomised from what the challenge's input can show without labels.

### Cited Findings
- An imperfect-tracking study argues that i.i.d. Gaussian noise misses tracker error structure. It fits a density to tracker residuals to draw each person's scale, then runs an OU process (θ = 0.15). On pedestrian trajectory forecasting, error was lower than with i.i.d. noise for all three backbones, for example VISTA minADE 0.64 against 0.92 — [arXiv 2608.30899](https://arxiv.org/html/2608.30899). Pedestrians, not football, and how θ was set is not stated.
- Sports tracklets show mix-up errors (one tracklet spanning players) and cut-off errors (a new ID for the same player) — [GTA](https://arxiv.org/pdf/2411.08216).
- Pan-identity switches from leaving the broadcast view were the main error source in hockey — [arXiv 2205.10949](https://arxiv.org/pdf/2205.10949).
- Three degradation draws and mixed gaps were the largest measured gains so far, at +0.066 / +0.087 — [BITACORA.md](../../BITACORA.md).

### Inferences
Calibration with no labels, from the challenge input only:
- **Identity swaps:** rate of implausible one-frame jumps per tracklet, and track-end to track-start distances within groups.
- **Missing players:** per-frame count of detected people, conditioned on nearby density.
- **Jitter:** speed spectrum of near-stationary players at stoppages, which gives both scale and correlation time, so θ can be estimated rather than assumed.
- Whether using input statistics like these is allowed depends on [RULES.md](../../RULES.md), and the earlier notes flagged the same check.

Validate by adding one more degradation mode as a draw, then scoring paired IDSSE leave-one-out and Metrica. Since three draws were the cheapest gain so far, a fourth draw with OU noise is the most promising model-side arm. Gain is unknown. The earlier discriminator check would show whether realism actually improved.

### Gaps
- No football-specific noise calibration study was found. No published identity-swap rate for automatic Metrica tracking was found.
- The 2026 Sports Engineering comparison noted earlier remains unread.

## Calibration, ensembling and decision theory for spellF1 under definition uncertainty

### Takeaway
Treat definition as a two-scenario decision and pick the decoder cost by expected regret. Further probability calibration is unlikely to matter, since ensembling and MBR add nothing beyond the frontier.

### Cited Findings
- The decision-theoretic prediction maximises expected utility under the estimated distribution, and estimate-then-decode can be near optimal with a reasonable model — [Jansche](https://www.academia.edu/52824743/A_maximum_expected_utility_framework_for_binary_sequence_labeling).
- F-measure maximisers can need the joint distribution — [Waegeman et al.](https://arxiv.org/abs/1310.4849).
- Direct utility tuning appears more robust to misspecification, while decision-theoretic methods do better with a good model — [Ye et al.](https://arxiv.org/abs/1206.4625).
- MBR with sampled paths scored −0.005 (1/7) — [BITACORA.md](../../BITACORA.md).
- Unsupervised scorers are unreliable for choosing settings — [SKADA-Bench](https://ar5iv.labs.arxiv.org/html/2407.11676).

### Inferences
- Tuning a single decoder constant across declared scenarios matches the "direct utility tuning" side, and suits a misspecified model. The frontier table in the first section is the decision table.
- The right objective is minimum expected regret across (raw, ctrl) × (uniform, dead), with π declared beforehand. Minimax regret on IDSSE falls near cost 24 (about 0.027, interpolated between grid points). Max-min score is the wrong rule.
- Ensembling across degradations is already in `mixed_all3`. Blending definitions moved along the frontier rather than beyond it.

**Ranked options**

| Rank | Option | Evidence | Under ctrl | Under raw-like | Validation |
|---|---|---|---:|---:|---|
| 1 | Switch cost 14–20 | **Measured**, IDSSE and Metrica | +0.023 to +0.074 | 0 to −0.023 | Rerun in production pipeline, freeze before any new look |
| 2 | Extra OU degradation draw | Pedestrian study; repo draws gain | Unknown | Unknown | Paired IDSSE + Metrica |
| 3 | Boundary-jittered labels | Video only | Unknown | Unknown | Paired, ≥ 0.01 on both |
| 4 | Per-match normalisation or CORAL | Mixed tabular | Unknown, probably small | Unknown | Metrica, fixed settings |
| Drop | Saerens or oracle priors | **Measured** no gain | ≤ +0.004 | ≤ 0 | Done |
| Drop | Duration merging | **Measured** dominated | | | Done |
| Drop | Raw + ctrl blend | **Measured** no hedge gain | | | Done |

### Gaps
- No method for expected temporal-IoU decoding under a mixture of label definitions was found.
- A blend tested only one ctrl draw, against three raw draws, so a three-draw ctrl model might sit slightly above the frontier.
- A dead-cost sweep and a per-state switch cost were not measured.
