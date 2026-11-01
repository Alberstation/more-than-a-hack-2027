# Inferring the team in possession without the ball: candidates for the H/A decision (state at 2026-11-01)

Scope: replace or augment the H vs A decision, the known limit (spellF1 0.48 now, 0.909 with true team, 0.516 with true dead ball). Code claims below come from reading the repos on 2026-11-01, not from READMEs alone. Nothing here was executed. Runtime figures for the RTX 4060 are my estimates, and are labelled as such.

Three corrections to the earlier notes in `research_notes/Posesión de balón en tracking/`:
- PathCRF **does** report frame-level team possession: 92.40%. The earlier note said it gave only the aggregate share.
- PathCRF and Ball Radar are **not** identity-free. They are invariant to slot order within a frame, but they run a temporal model per slot, so a slot must be the same player across the 10 s window (see Q1).
- The earlier suggestion that a 41% missing ball is "close to" Ball Radar's 80%-masked row is too optimistic. Our gaps are contiguous, with a median of 10.7 s, about the length of its 10 s window (see Q2).

## Q1. PathCRF, Ball Radar and newer models: real numbers, code state, tracklets, IDSSE retraining

### Takeaway
PathCRF is the only public candidate with weights, a permissive-enough licence and numbers on our own dataset family. Its 92.40% team accuracy is the best ball-free figure found, but it comes from **one match of clean data with 22 identified players**, and it falls sharply across providers (event F1 73.9 to 49.2). It should be treated as a ceiling to measure first, not as a drop-in replacement. Nobody has published a result on fragmented, anonymous tracklets.

### Cited Findings

**Candidate table**

| Candidate | Reported result (data) | Code / licence / last commit | Weights | Independently reproduced? | Input needs | Effort for tracklets | RTX 4060 (my estimate) | spellF1 relevance |
|---|---|---|---|---|---|---|---|---|
| [PathCRF](https://arxiv.org/html/2602.12080v2) (KDD'26) | Team 92.40% per frame, event F1 75.69, edge 69.64 (1 Sportec test match); leave-one-match-out F1 73.88 ± 1.90 | [repo](https://github.com/hyunsungkim-ds/pathcrf), MPL-2.0, 2026-06-08, 26 stars | Yes, 5 configs, ~19 MB each | No. Same authors only | 22 fixed slots + 4 outside nodes, team sign, GK flag, x, y, vx, vy, speed, accel. No ball. No NaN allowed | High: lanes needed, or rewrite | Inference ~1-3 min/match. Dynamic CRF about 20+ h per 50-epoch run; non-CRF about 8 h | Highest potential, unproven on our input |
| [Ball Radar](https://arxiv.org/abs/2306.08206) (KDD'23) | Ball error 3.66 m, player possessor 64.7%, team 85.85% (Metrica plus private GPS, values from earlier note) | [repo](https://github.com/hyunsungkim-ds/ballradar) (moved from pientist), **no root LICENSE (HTTP 404)**, pushed 2026-02-03, 35 stars | **None in the tree** | Partly: rivals re-ran it, with very different numbers (below) | Fixed players per team; `team_ball.sh` uses `n_players 10`, `n_features 6`. Ball optional for imputation | High | Bi-LSTM, likely similar to non-CRF PathCRF | Medium. Superseded by PathCRF; licence risk |
| [TranSPORTmer](https://arxiv.org/html/2410.17785v2) (ACCV'24) | Ball inference 2.71 m, state accuracy 80.84% (283 LaLiga matches, 6.25 Hz, proprietary) | No official code found. Rival reimplementation: [`models/transportmer.py`](https://github.com/hyunsungkim-ds/pathcrf) (MPL-2.0) | No | Contradicted on Sportec (below) | 22 + ball with type; **NaN mask built in**, so missing agents are handled by design | Medium, if rebuilt | Not measured | Medium for the NaN-mask idea only |
| [Peral et al.](https://arxiv.org/abs/2512.19528) (WACV'26) | Full model possessor 87.67%, trajectories only 82.12% (283 matches, proprietary) | No repo in the paper | No | No | Trajectories, team type, **video crops** | Not viable: crops rejected for us, no code | n/a | Low |
| [Borghesi et al.](https://iris.polito.it/handle/11583/2976470) (ESWA'23) | 86.2% possession accuracy, +7.3 pts over rules ([Serie A 2019-20 per the thesis](https://webthesis.biblio.polito.it/18073)) | No code found; thesis embargoed | No | No | Dead/home/away classes; permutation-invariant layers per a search summary of the paper. **Ball use unverified** | Design is reusable | Light | High as a template, not as an asset |
| [Baouan et al.](https://arxiv.org/pdf/2501.10299) OT embedding | 81.66% (logistic), 82.26% (MLP) from one frame of positions (Ligue 1, Stats Perform, 100 games) | No GitHub link in the PDF | No | No | Positions per frame, no ball, no IDs needed in the invariant variants | Low | CPU | Cheap feature source; see Q3 caveats |
| [football-possession-estimator](https://github.com/saideepojha/football-possession-estimator) | LightGBM + Viterbi 0.964/0.971, 407/327 flips on Metrica samples | MIT | n/a | No. Author calls it a development signal | Uses a nearest-player rule; undetermined frames excluded | Low | CPU | Benchmark for flips only; not comparable |

**PathCRF, in detail**
- Sportec split: 5 train, 1 validation, 1 test; 5 fps, 10 s windows. Test match: edge 69.64, sender 80.44, receiver 82.28, event P/R/F1 73.18/78.37/75.69, with 1,961 predicted events against 1,831 true. Team possession frame accuracy 92.40%; home share 62.64% against 62.13%. — [arXiv v2](https://arxiv.org/html/2602.12080v2)
- Appendix B, mean ± sd over test matches: — [arXiv v2](https://arxiv.org/html/2602.12080v2)

  | Train → test | Edge acc. | Event F1 |
  |---|---|---|
  | Sportec → Sportec (leave-one-match-out) | 66.75 ± 2.76 | 73.88 ± 1.90 |
  | K League → K League | 68.20 ± 2.64 | 79.83 ± 2.60 |
  | K League → Sportec | 52.71 ± 9.31 | 66.25 ± 7.65 |
  | **Sportec → K League** | **41.65 ± 2.34** | **49.16 ± 3.37** |

- Decoding ablation on event F1: no CRF 65.34; no CRF with Viterbi constrained decoding 73.14; static masked CRF 73.83; dynamic masked CRF 75.69. — [arXiv v2](https://arxiv.org/html/2602.12080v2)
- 25 fps gives F1 77.21 at about 12x the training time. Timings on an RTX 4090: 184 s/epoch non-CRF, 242 static, 565 dynamic; about 20-50 s per match. — [arXiv v2](https://arxiv.org/html/2602.12080v2)
- The team figure is derived from events. [`applications/team_poss.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/main/applications/team_poss.py) labels in-play frames from consecutive event teams and defaults to mode "split", which splits a team-change gap at its midpoint. So the 92.40% seems to be measured against an event-derived timeline, not the DFL `ball_owning_team` feed. I did not confirm from the paper which mode was used.
- Code facts from [`dataset.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/main/dataset.py) and [`models/set_lstm.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/main/models/set_lstm.py):
  - Node order is fixed as left team, right team, then 4 outside nodes.
  - Any episode with a NaN in a node feature is **skipped**.
  - Features are reshaped to `(T, B*N, Z)` and passed to a temporal model per node, which is what requires slot continuity across the window.
  - `team_size` is fixed in the CRF edge maps.
- `saved/140/args.json` shows `cv_fold_idx: 6`, `node_in_dim: 8`, 3.43M parameters, batch 32, 50 epochs, seed 100. — [args.json](https://github.com/hyunsungkim-ds/pathcrf/blob/main/saved/140/args.json)
- `train.py` accepts `--macro_type poss_team`, and the dataset builds a team target from `ball_owning_team_id`. That target is the macro output, and I did not check that the team-only path trains end to end. — [train.py](https://github.com/hyunsungkim-ds/pathcrf/blob/main/train.py)
- [`datatools/preprocess_noise.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/main/datatools/preprocess_noise.py) defines occlusion, bias and outlier noise, but `SCENARIOS = []` and the paper's main text reports no noise experiment. It has no player-dropout scenario. Treat this as abandoned or unreported work.
- [`datatools/metrica_data.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/main/datatools/metrica_data.py) exists, so a Metrica parser ships with the repo, but the paper's reported datasets are Sportec and K League. The repo also has `models/gat_lstm.py`, with no matching result that I saw.
- `requirements.txt` is unpinned and lists no torch or torch_geometric, though `dataset.py` imports both. Environment reproduction will need work. — [requirements.txt](https://github.com/hyunsungkim-ds/pathcrf/blob/main/requirements.txt)

**Ball Radar**
- Root LICENSE returns 404; only `set_transformer/LICENSE` exists. No `.pt` or `.pth` files in the tree. — [repo](https://github.com/hyunsungkim-ds/ballradar)
- Imputation, position error / possessor accuracy before post-processing: 100% masked 5.35 m / 61.9%; 95% 3.00 / 79.5; 90% 2.09 / 87.1; 80% 1.31 / 93.1. Values come from the earlier note, citing [arXiv](https://arxiv.org/html/2306.08206v1).

**Reproduction evidence, which is the most useful negative result here**
- Ball Radar error depends on who runs it: 3.66 m (own), [3.89 m](https://arxiv.org/html/2410.17785v2) (TranSPORTmer authors), [1.86 m](https://arxiv.org/pdf/2512.19528) (Peral), [2.74 m](https://arxiv.org/html/2602.12080v2) (PathCRF "Ball TP", built on Ball Radar heuristics).
- TranSPORTmer itself moves from [2.71 m](https://arxiv.org/html/2410.17785v2) to [1.72 m](https://arxiv.org/pdf/2512.19528) between two papers from the same lab. Peral says it discards some sequences, which may explain part of this.
- TranSPORTmer's "best" claim does not survive on Sportec. Backbone ablation, event F1 / edge acc: SAB + PE-SABs (TranSPORTmer) 52.34/58.49 no CRF, 68.19/62.97 with CRF. PPE-FPE + Bi-LSTM (Ball Radar) 66.77/69.15 and 75.24/70.39. Proposed 65.34/69.24 and 75.69/69.64. — [arXiv v2](https://arxiv.org/html/2602.12080v2). Caveat: a rival's reimplementation, trained on 5 matches rather than 283.

### Inferences
- The Bi-LSTM backbone is about as good as the attention one on Sportec (75.24 against 75.69). With 7 training matches there is no evidence for a heavier model.
- Most of the structured gain comes from constrained Viterbi (65 to 73), which we already use in a 3-state form. Moving from 3 states to a richer state space may help; replacing Viterbi with a CRF probably adds little more.
- Sportec → K League is the closest published analogue to IDSSE → challenge match, because the provider changes. Expect leave-one-match-out on IDSSE to overstate what we get on the target. This applies to our current model too.
- Do not use the shipped weights across our cross-validation. They are trained on matches we also evaluate on, and only the fold-6 held-out match is clean for those weights. I only checked this for trial 140.
- Runtime arithmetic: I assume a 4060 is about 3x slower than a 4090 for this workload (assumption, not measured). Dynamic CRF would be about 565 × 3 × 50 s ≈ 24 h per fold, so full leave-one-match-out is out of reach before 17 Nov. Non-CRF would be about 8 h. A larger window stride would cut this. 8 GB at batch 32 is untested.

### Gaps
- Which match is fold 6, and the exact mode behind 92.40%. Both can be resolved from `saved/140/log.txt` and `tutorial.ipynb`, which I did not read.
- No paper reports turnover-level recall, per-spell metrics, or team accuracy across matches. PathCRF's 92.40% has no variance estimate.
- Ball Radar's `team_ball` variant has `--target_type gk`, whose meaning I did not trace.
- Borghesi full text was blocked (403, or a JavaScript wall on the accepted manuscript), so dataset size, ball use and any split are unverified.

## Q2. Permutation-invariant, set and graph models; multi-agent imputation

### Takeaway
There is no tested, public, identity-free possession model. The design we want is a set encoder per frame, pooled by team, followed by a temporal model, and it has precedent only in pieces. Imputation models target players or a mostly observed ball, and neither matches contiguous 10.7 s gaps.

### Cited Findings
- **TranSPORTmer**: alternating temporal and social set attention, with a NaN mask for missing agents and padding. Goalkeepers with 97-100% hidden observations are used in training. Its CLS token classifies pass/possession/uncontrolled/out. Per-class accuracy appears only in figures; the text says uncontrolled is the weakest class. — [arXiv v2](https://arxiv.org/html/2410.17785v2)
- Imputation (ADE): Ball Radar 0.88 m at 80% masked and 1.18 m at 90%; TranSPORTmer about 0.84 and 1.09. Table row alignment is partly unverified because the HTML was flattened. — [arXiv v2](https://arxiv.org/html/2410.17785v2)
- **Peral et al.**: trajectories only 1.97 m, state 77.79%, possessor 82.12%. Adding team type gives 1.78 / 79.64 / 83.66, and the full model 1.45 / 84.38 / 87.67. Baselines: Ball Radar possessor 78.12, TranSPORTmer 83.79. Matches are disjoint across splits. — [arXiv PDF](https://arxiv.org/pdf/2512.19528)
- **UniTraj / Sports-Traj** (ICLR 2025): masked-input player trajectory recovery; Apache-2.0, 28 stars, checkpoints on Google Drive, needs `mamba-ssm`. Soccer data is SoccerTrack top-view scenarios. The repo page does not confirm ball support. — [repo](https://github.com/colorfulfuture/UniTraj-pytorch), [arXiv](https://arxiv.org/html/2405.17680v2)
- **baller2vec**: MIT, last commit 2021-08-30, 71 stars. Its tasks are basketball trajectory forecasting, not possession inference. — [repo](https://github.com/airalcorn2/baller2vec), [paper](https://ar5iv.arxiv.org/html/2102.03291)
- **MIDAS**: player imputation, not ball; it reportedly beats ImputeFormer among others, per a secondary summary not verified. — [arXiv](https://arxiv.org/html/2408.10878v4)
- **Off-screen player imputation** (2026): training-free imputation of hidden players, tested by simulating a viewport over Metrica full-pitch data. Search snippet only. — [arXiv 2607.11548](https://arxiv.org/pdf/2607.11548)
- **GNNs**: no paper with a team possession result was found. TGNet is a blog summary, and its possession task and numbers were not checked. — [Substack](https://thexgfootballclub.substack.com/p/turning-football-tracking-data-into)

### Inferences
- A frame-level set encoder sidesteps the lane problem completely: encode each visible player with x, y, vx, vy and team, pool per team (mean plus attention), then run a TCN or Bi-LSTM over pooled frames. It takes 20.5 players or 46 fragmenting slots without stitching. This is the most promising new model, and it is my proposal, not a validated method.
- A team-only loss also avoids a real labelling problem. Edge labels need the true possessor to be present in the degraded input, which fails when that player is dropped. A team label does not.
- Imputation rows are the wrong guide for us. In a 10.7 s median gap, the middle of a 10 s window has almost no ball, so the 100%-masked row (5.35 m, 61.9%) is closer than the 80% row. Only gap edges benefit. This assumes those papers mask scattered frames or short runs, which I did not confirm.
- Filling in the ball and then deciding H/A is also disfavoured directly: "Ball TP" reached event F1 48.39 against 75.69 for direct path inference, though at 2.74 m ball error. Predict the team directly.
- Baller2Vec, UniTraj and ImputeFormer-type models are not worth deadline time for H/A. None has a possession result.

### Gaps
- Mask patterns in Ball Radar and TranSPORTmer imputation (scattered or contiguous) were not confirmed.
- I did not verify whether baller2vec depends on player identity embeddings.
- No study measures any of these under tracklet fragmentation.

## Q3. Which signals identify the team in possession without the ball

### Takeaway
Shape has real evidence, but at long time scales. Team-level spread and relative geometry reach about 80-82% per frame in one study, while mean position alone gives about 60%. All of this evidence is for sustained phases, and nothing found shows that shape can flag a possession under 2 s. I found no study of velocity alignment as a possession classifier.

### Cited Findings
- Frame-only possession prediction: mean positions 59.31 / 59.92; raw tracking 59.65 / 76.77; 10×10 occupancy grid 73.71 / 80.03; embedding 81.66 / 82.26; centred embedding 81.81 / 80.63 (logistic / MLP or CNN). — [Baouan et al.](https://arxiv.org/pdf/2501.10299)
- Caveats on that table:
  - 64,024 frames at 1-in-10 subsampling, with unassigned frames discarded.
  - The split is described only as cross-validation and a "train-test split", not by match. Frames 0.4 s apart from one game could sit on both sides, which would inflate results.
  - The prose quotes 60.33% and 59.47% where the table shows 59.65 and 59.31.
- Shaw & Glickman, 180 matches at 25 Hz: formations are measured in windows with **at least one minute** of in-play data, separately in and out of possession. In a one-team example the outfield convex hull when attacking was twice that when defending, and a team moved from a 4-1-4-1 out of possession to a front three in possession. — [PDF](https://www.sportperformanceanalysis.com/s/Dynamic-analysis-of-team-strategy-in-professional-football-By-Laurie-Shaw-And-Mark-Glickman.pdf)
- Spanish league AMISCO data, 6,793 possessions: width, depth and surface area of the playing area were larger in possession. Search summary, full text not read. — [Revista de Psicología del Deporte](https://ddd.uab.cat/pub/revpsidep/revpsidep_a2013v22n2/revpsidep_a2013v22n2p437.pdf)
- One-match case study, also from a search summary: centroid and surface area differ with and without possession. — [record](https://lida.sport-iat.de/dfb/Record/4030810?lng=en)
- Pressing and compactness papers study defensive success, not who has the ball. A scoping search found work on local compactness near the ball, but it needs the ball location. — [arXiv 2511.06191](https://arxiv.org/html/2511.06191v1)
- PathCRF learns from x, y, vx, vy, speed, accel, team and GK flag, and publishes no feature attribution. — [dataset.py](https://github.com/hyunsungkim-ds/pathcrf/blob/main/dataset.py)
- Our own features in [`possession/features.py`](../../possession/features.py), collective-shape block: team centroid x, x and y spread, mean speed, centroid difference. That block has no hull area, projections, occupancy grid or signed team velocity. I read only that block, lines 170-216, so other parts of the file may hold related signals.

### Inferences
- The ~60% for mean position against ~80% for occupancy or embedding suggests that our centroid and spread features may be missing most of the available shape signal, if those numbers hold under match-level splits.
- Cheap additions to test, all invariant to identity and tolerant of 20.5 players:
  - team hull area and asymmetry;
  - sorted x and y quantiles per team, and on a few rotated axes, as an approximation of the sliced-Wasserstein embedding;
  - a coarse occupancy grid per team, attack-normalised;
  - signed team velocity toward goal, and its change over 1 s;
  - distance from each team's players to their nearest opponent.
- Physical reasoning: a team cannot reshape in under 2 s. So shape can only retrospectively support a sustained possession. It should improve long-spell H/A choice in gaps, but it will not by itself find the 30% of possessions under 2 s. Those matter to the 437 against 280-300 shortfall.
- A short-spell signal would need to be local and fast, for example sudden direction changes of several players at once, or a run toward a candidate receiver. This is an open bet with no evidence behind it.
- Mean possession length is about 7.5 s if in-play time is near 55 minutes (assumption). PathCRF's 7.6% frame error would then average about 0.6 s per spell, which would be compatible with boundary lag and good IoU, but equally with a few long swapped spells. So 92.4% says little about spellF1.

### Gaps
- No published comparison of signals under match-level holdout.
- No evidence on latency: how fast shape features react after a turnover.
- No study of velocity alignment or pressing as a direct possession classifier, despite targeted searches. This is a negative finding, not proof of absence.

## Q4. What is validated, what is only claimed, and what to do before 17 Nov

### Takeaway
No method is validated for our conditions, and no number in the literature maps to spellF1. The sensible path is to measure PathCRF's ceiling with our metric in about a day, add cheap invariant features, then build an identity-free team model only if the ceiling justifies it. Full PathCRF retraining with lanes should come last.

### Cited Findings

| Claim | Status |
|---|---|
| PathCRF 92.40% team accuracy | Authors only, one match, clean data — [arXiv v2](https://arxiv.org/html/2602.12080v2) |
| PathCRF event F1 about 74-76% on Sportec | Authors only, but with leave-one-match-out; weights are public, so checkable — [arXiv v2](https://arxiv.org/html/2602.12080v2) |
| "Near-perfect" possession share | Aggregate, errors can cancel; not relevant to spells — [arXiv v2](https://arxiv.org/html/2602.12080v2) |
| Robustness across providers | **Refuted in the paper's own appendix**: 73.88 to 49.16 — [arXiv v2](https://arxiv.org/html/2602.12080v2) |
| Robustness to noise or missing players | Not tested; noise code is present but disabled — [preprocess_noise.py](https://github.com/hyunsungkim-ds/pathcrf/blob/main/datatools/preprocess_noise.py) |
| TranSPORTmer beats Ball Radar | True on its own data; reversed on Sportec under rival reimplementation — [arXiv v2](https://arxiv.org/html/2602.12080v2) |
| Ball Radar works on Metrica | Authors only, with test IDs reassigned by ball distance in one match, per earlier note — [arXiv PDF](https://arxiv.org/pdf/2306.08206) |
| 82% from position embeddings | Single study, split possibly leaky — [arXiv PDF](https://arxiv.org/pdf/2501.10299) |
| 96-97% with LightGBM on Metrica | Excludes undetermined frames, event-derived labels; the author warns of coupling — [repo](https://github.com/saideepojha/football-possession-estimator) |

### Inferences
Ranked plan, with today at 1 Nov. Effort figures are my estimates.

1. **PathCRF ceiling test, about 1 day.** Run the shipped trial 140 weights on the fold-6 held-out match with clean IDSSE input, turn predictions into H/A/D spells, and score them with our spellF1. In the same run, score our current model on that match, with degraded and clean input.
   - Go/no-go: if clean, identified, complete input does not clearly beat our 0.48, degraded input will not, so drop the deep route.
   - If it does beat it, compare how many short spells it recovers.
2. **Invariant shape features in the HGB, about 1-2 days.** Use the list in Q3, under match-level folds. Lowest risk and fully inspectable. Expect a gain on long gaps, not on short spells.
3. **Identity-free team model, about 3-4 days, only if step 1 passes.** Frame set encoder, team pooling, Bi-LSTM or TCN at 5 fps, team-only loss, trained with the degradation already in `possession/train.py`, and fed as log-odds into the existing Viterbi. Ball as an optional input with an observed flag. Fix seeds and use deterministic settings. Training should fit overnight runs on the 4060 (estimate).
4. **PathCRF retrain with stitched lanes, last resort.** Assign tracklets to 22 lanes per window, add a validity flag and replace the NaN skip. Needs GK detection on our data, about 1 week or more, and each dynamic run takes about a day. Non-CRF with constrained Viterbi is the only sensible variant.

- Use Metrica sample matches as a cross-provider holdout for steps 2 and 3, because IDSSE-only folds will overstate generalisation. Labels there are event-derived, so results are approximate.
- Licences: PathCRF is MPL-2.0, which is fine; changes to its files stay MPL. Avoid copying Ball Radar code. UniTraj is Apache-2.0, baller2vec MIT.
- All candidates run locally, so the no-cloud constraint is met. No candidate requires labels on the challenge match.
- Expected relevance: I would not forecast a spellF1 number. The evidence supports a gain on long gaps, and leaves short possessions unresolved.

### Gaps
- No external benchmark for spell-level possession.
- Whether the challenge's Metrica automatic tracking behaves more like IDSSE or something else cannot be judged from the literature.
- Metrica sample data licence was not checked in this pass.
- Our current H vs A frame accuracy inside gaps was not found in the docs I searched, and it is needed to compare with 92.40%.
