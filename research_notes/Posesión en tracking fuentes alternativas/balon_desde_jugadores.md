# Inferring the ball, ball carrier or possessing team from player trajectories

Checked 2026-11-01. Nothing downloaded was executed. Repo facts come from the GitHub API or from earlier notes where stated. Effort, size and runtime figures labelled "estimate" are mine.

**Bottom line:** the project's own dose-response is a stronger constraint than anything in the literature. A learned ball *coordinate* is unlikely to reach +0.03 alone. The options left are a direct team posterior with ball location as an auxiliary target, and cheap ball-proxy features. See Q4.

New since the earlier notes:
- TranSPORTmer, UniTraj and Peral feed attacking/defending roles as input.
- Peral excludes out-of-play sequences.
- Ball Radar's team accuracy under masking.
- UniTraj's mask definitions.
- U2Diff and U2Diffine.
- SoccerTrack v2's ball-free baseline and its numbers.
- FOOTPASS positions as no-ball training data.
- T-GAN by Li & Link.
- The leave-two-matches-out stacking scheme.

## Q1. State of the art in ball inference and imputation from multi-agent context

### Takeaway
The field runs in three steps: Ball Radar (Set Transformer + Bi-LSTM), then TranSPORTmer (set attention), then Peral (adds video crops), with a diffusion branch (U2Diff, U2Diffine). PathCRF moves away from ball coordinates to direct possession paths. No public code exists for TranSPORTmer, Peral or U2Diff that I could find. Best trajectory-only errors are about 2-3 m on clean, complete, in-play data, and none is measured on anonymous degraded input.

### Cited Findings

**Candidate table** (only the rows marked "new" were absent from the earlier notes)

| Option | What it does | Input needs | Measured result and data | Code, licence, last commit | Size | Training on degraded IDSSE |
|---|---|---|---|---|---|---|
| [Ball Radar](https://arxiv.org/html/2306.08206v1) (KDD'23) | Set Transformer, hierarchical Bi-LSTM: possessor first, then ball | Fixed player slots per team across a 10 s window; ball optional | 100% masked: 5.35 m, player 61.91%, team 83.96%. 95%: 3.00 m, 79.51%, 91.54%. 90%: 2.09 m, 87.11%, 95.04%. 80%: 1.31 m, 93.13%, 97.41% ([3 Fitogether GPS matches, 10 Hz](https://arxiv.org/html/2306.08206v1)) | [repo](https://github.com/hyunsungkim-ds/ballradar): no root licence, no weights, per [earlier note](../Posesión%20en%20tracking%20cinco%20mejoras/equipo_sin_balon.md); not rechecked | [8,699,846 parameters](https://arxiv.org/html/2306.08206v1) for the full ablation model, probably this one | Blocked by licence and per-slot temporal model |
| [TranSPORTmer](https://arxiv.org/html/2410.17785v2) (ACCV'24) | Alternating temporal and social set attention; CLS token for state | Agent type (ball, offensive, defensive), x, y; 60 frames at 6.25 Hz | 2.71 m, MaxErr 7.39 m, state 80.84%; masked 80%/90%: 0.84/1.09 m ([LaLiga 2022-23, proprietary](https://arxiv.org/html/2410.17785v2)) | No code URL in paper. Rival reimplementation in [PathCRF repo](https://github.com/hyunsungkim-ds/pathcrf), MPL-2.0 | [Not reported](https://arxiv.org/html/2410.17785v2): 128 dim, 16 heads, 512 hidden | Rebuild with H/A type, not attack/defence |
| [Peral et al.](https://arxiv.org/html/2512.19528) (WACV'26) | Masked pre-training; joint ball, state, possessor | x, y, offensive/defensive type, video crops | Trajectories only 1.97 m, state 77.79%, possessor 82.12%. Full 1.17 m, 85.85%, 89.56% (283 matches, proprietary) | [No code URL](https://arxiv.org/html/2512.19528) | Not reported | Not viable: needs crops |
| **New:** [U2Diff](https://arxiv.org/abs/2503.18589v2) / [U2Diffine](https://arxiv.org/html/2605.10717) (CVPR'25, TPAMI) | Diffusion trajectory completion with per-state uncertainty; 6 DDIM steps | x, y, mask, [learnable per-agent embedding](https://arxiv.org/html/2605.10717); ball is one agent | minADE20 Soccer-U 53.93 and 52.89 pixels, [not split by ball](https://arxiv.org/html/2605.10717) | [No code repository](https://arxiv.org/abs/2605.10717) | Not reported; [8-59 ms per mode at batch 128, RTX A6000](https://arxiv.org/html/2605.10717) | Reimplementation only |
| [UniTraj](https://arxiv.org/html/2405.17680v2) (ICLR'25) | Mamba-based masked trajectory generation | [Agent category: ball, offensive, defensive](https://arxiv.org/html/2405.17680v2) | minADE20 Soccer-U 94.59 pixels, [not split by ball](https://arxiv.org/html/2405.17680v2) | [repo](https://github.com/colorfulfuture/UniTraj-pytorch), [Apache-2.0, pushed 2025-03-17, 28 stars](https://api.github.com/repos/colorfulfuture/UniTraj-pytorch) | [About 1.77M params](https://arxiv.org/html/2405.17680v2) | Possible, poor fit. Replace type with H/A; needs `mamba-ssm` per earlier note |
| [PathCRF](https://arxiv.org/html/2602.12080v2) (KDD'26) | Direct possession paths, no ball | 22 slots, GK flag, no NaN | Team 92.40%, 1 Sportec match | MPL-2.0, [pushed 2026-06-08, 26 stars](https://api.github.com/search/repositories?q=ball+inference+player+trajectories&per_page=15) | 3.43M, per earlier note | Blocked locally: [no external code run; needs identities and GK](../../BITACORA.md) |
| **New:** [Amirli & Alemdar](https://dergipark.org.tr/en/pub/apjess/article/1060725) (2022) | Neural net predicts ball x, y from player positions | Player spatial positions; ordering unverified | MAE 7.56 m x, 5.01 m y, R² 79%/92% (300 Turkish Super League matches; abstract only) | Not found | Not found | Simple baseline to imitate |
| **New:** [SoccerTrack v2 baseline](https://arxiv.org/html/2508.01802v2) | TCN over 82 aggregate per-frame features, ±25 s field; 12 ball action classes | Aggregate shape, density grids, "soft minimum over opposing pairs as a ball proxy". All 22 ground-truth players as published | mAP trajectory-only 0.222 (±1 s), 0.524 (±5 s); with ball 0.599, 0.666. 10 matches, 5 folds | [Tooling MIT per paper](https://arxiv.org/html/2508.01802v2); [GitHub API says NOASSERTION, pushed 2026-10-05, 41 stars](https://api.github.com/repos/AtomScott/SoccerTrack-v2) | Not reported | Features reusable directly |
| **New:** [Li & Link T-GAN](https://arxiv.org/html/2606.09289v1) (preprint 2026) | GNN per frame, attention pooling by team and role, Transformer | TRACAB positions, players and **ball**, action and possession context; identities not stated | Macro F1 0.87 frame; sequence F1 0.79 after post-processing, 0.68 before (7 Bundesliga matches, leave-one-match-out). The task is phases within possession, not H/A | [No code URL](https://arxiv.org/html/2606.09289v1) | Not reported | Architecture precedent only |
| [Ochin et al.](https://arxiv.org/html/2502.15462v1) (2025) | Player GNN + 3D CNN for action detection | Position, velocity, team bit, visual feature; tracklet IDs link time | About 30% fewer false positives; GNN mAP only in figures (Footovision 997 games, private) | [No code URL](https://arxiv.org/html/2502.15462v1) | Not reported | Not viable: needs video |
| **New as data:** [FOOTPASS](https://arxiv.org/html/2511.16183v1) | 54 matches, 102,992 on-ball events, positions and velocities per frame | Team, jersey, role provided. [Off-screen players imputed](https://arxiv.org/html/2511.16183v1); no ball described | Not a model result for our task | [Baselines repo](https://github.com/JeremieOchin/FOOTPASS): [API Apache-2.0, pushed 2026-06-17, 31 stars](https://api.github.com/repos/JeremieOchin/FOOTPASS); [annotations CC BY-NC 4.0](https://raw.githubusercontent.com/JeremieOchin/FOOTPASS/main/README.md) | n/a | Extra no-ball training set, see Q4 |
| [Event2Tracking](https://ojs.aaai.org/index.php/AAAI/article/view/33289) (AAAI'25) and [Hughes et al.](https://researchr.org/publication/HughesLHGFS26/authors) (CVIU 2026) | Reconstruct trajectories over 60 s windows from tracking plus events | Event stream required, per the abstract | "Improves" player and ball accuracy; numbers not read | Not checked | Not read | Probably out of scope: no event stream on the challenge match |

**Access for gated sources**
- FOOTPASS videos need the [SoccerNet NDA form](https://raw.githubusercontent.com/JeremieOchin/FOOTPASS/main/README.md), which gives a password for extraction. The README's steps may also apply that password to annotation files. Dataset listed as SN-PCBAS-2026 on Hugging Face.
- SoccerTrack v2 is on [Hugging Face after accepting terms](https://huggingface.co/datasets/atomscott/soccertrack-v2), CC BY 4.0 per [the paper](https://arxiv.org/html/2508.01802v2). An [earlier note](../Posesión%20de%20balón%20en%20tracking/deteccion_balon_video.md) said only part of the per-frame annotation was released; confirm what the current release holds.
- Hughes et al. and Borghesi et al. returned 403 here. Use an institutional library or ask the authors. Lang et al. offer [data on reasonable request](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/), though it is unlikely to arrive in time.

**Comparability warnings**
- Ball Radar's error differs between runners: [3.66, 3.89, 1.86 and 2.74 m per earlier note](../Posesión%20en%20tracking%20cinco%20mejoras/equipo_sin_balon.md). TranSPORTmer reads 2.71 m in its paper and [1.72 m in Peral](https://arxiv.org/html/2512.19528).
- Peral's "ground-truth" ball is [interpolated from possessor annotations](https://arxiv.org/html/2512.19528), and [sequences with out-of-play frames are excluded](https://arxiv.org/html/2512.19528). So those errors are for an idealised ball in live play.
- The earlier note gave Ball Radar team accuracy as 85.85%; the imputation table here gives [83.96% at 100% masked](https://arxiv.org/html/2306.08206v1). I did not reconcile them; they may be different experiments.

### Inferences
- With 7 matches, heavy attention or diffusion has no evidence in its favour. PathCRF's backbone ablation already showed the Bi-LSTM level with attention on Sportec.
- Diffusion is a poor fit for byte-identical CPU inference: 20 modes × 6 steps over a whole match, plus seeded sampling, with no public code. I would not build it.

### Gaps
- No ball-only ADE on Soccer-U, and its frame rate is not given.
- Li & Link's last 41k characters were not read, and I could not read Amirli's full text (PDF 404).
- Parameter counts are missing for most entries.

## Q2. Which work without identities and tolerate missing or swapped players?

### Takeaway
No published ball-inference model is demonstrated under identity swaps or fragmentation. "Permutation-equivariant" only means slot order within a frame. Most still need one row to be one agent across the window. Only aggregate-feature models are immune by construction, which is the family HGB already belongs to.

### Cited Findings
| Model | Identity need | Missing agents | Swap tolerance |
|---|---|---|---|
| TranSPORTmer | [No persistent identity; type only. Temporal axis per agent; agent order standardised for soccer](https://arxiv.org/html/2410.17785v2) | NaN mask; goalkeepers trained with [97-100% hidden](https://arxiv.org/html/2410.17785v2) | Not tested |
| U2Diff | [Learnable per-agent embedding](https://arxiv.org/html/2605.10717) | Mask | Slot-dependent, by my reading |
| UniTraj | [Category only](https://arxiv.org/html/2405.17680v2) | Mask, plus 5 agents wholly masked | Not tested |
| Peral | [Not stated; N logits per frame](https://arxiv.org/html/2512.19528) | Crop placeholders; pre-training masks only | Not tested |
| PathCRF, Ball Radar | Slot continuity, per earlier note | PathCRF skips NaN | No |
| Ochin | [Tracklet IDs link time](https://arxiv.org/html/2502.15462v1) | Not described | Not tested |
| Lang et al. (in-game status) | [Ordered by match sheet](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/); 22 players | Not described | No |
| SoccerTrack v2 baseline | [Aggregate per-frame features](https://arxiv.org/html/2508.01802v2) | Tolerant by design, not tested | Immune by construction |

- **Role leak:** TranSPORTmer's [offensive/defensive type](https://arxiv.org/html/2410.17785v2), Peral's [team type](https://arxiv.org/html/2512.19528) and UniTraj's [category](https://arxiv.org/html/2405.17680v2) encode who attacks, which is close to what we must predict. In Peral, adding types moves trajectories-only from 1.97 m, 77.79%, 82.12% to [1.78 m, 79.64%, 83.66% per earlier note](../Posesión%20en%20tracking%20cinco%20mejoras/equipo_sin_balon.md).

### Inferences
- Published scores for these three are not attainable by us as stated: we can supply home/away, not attack/defend. Peral's ablation suggests the effect is moderate for ball error. Its possession-related effect is unknown because possessor accuracy is not team accuracy.
- A frame-level set encoder, pooled by team, then a temporal model on pooled vectors, is swap-immune because no temporal link crosses tracklets. T-GAN is new precedent for this pattern and worked on 7 matches, though with ball input and a different task.

### Gaps
- No study measures tolerance to the 20.5 visible players or the slot fragmentation of the challenge data.
- Whether TranSPORTmer's standardised order is needed by the model is [not stated](https://arxiv.org/html/2410.17785v2).

## Q3. Accuracy in long ball-absent gaps of 5-20 s

### Takeaway
No source reports accuracy by contiguous gap length in football ball estimation. The closest evidence is the fully masked ball over about 10 s: 2.7-5.4 m and about 84% team accuracy on clean data. Scattered-mask rows look far better, and do not apply.

### Cited Findings
- Ball Radar masks are ["randomly mask[ed] ... by a certain probability"](https://arxiv.org/html/2306.08206v1), with no contiguous-segment definition stated. Training uses 80% masked for half the batches and no ball for the other half.
- TranSPORTmer's ball masks at 80% and 90% [give no contiguity or gap length](https://arxiv.org/html/2410.17785v2).
- U2Diff uses [UniTraj's five masks](https://arxiv.org/html/2605.10717). UniTraj's [center consecutive mask](https://arxiv.org/html/2405.17680v2) is the only long-hole design found: one hole of 25-40 frames, or 4-6.4 s at 6.25 Hz. Scores are pooled across masks and agents, so no 5-20 s ball figure exists. Its random holes are 3-5 frames.
- Full-window ball-free figures: Ball Radar [5.35 m, team 83.96%](https://arxiv.org/html/2306.08206v1); TranSPORTmer [2.71 m, MaxErr 7.39 m](https://arxiv.org/html/2410.17785v2); Peral trajectories only [1.97 m, MaxErr 7.12 m](https://arxiv.org/html/2512.19528); Amirli [MAE 7.56/5.01 m per axis](https://dergipark.org.tr/en/pub/apjess/article/1060725).
- Time scale of the penalty: trajectory-only mAP is [0.222 against 0.599 at ±1 s, and 0.524 against 0.666 at ±5 s](https://arxiv.org/html/2508.01802v2). The with-ball scores may be inflated, because the ball track is reconstructed around events.
- Long imputation becomes [inherently harder, per MIDAS on player imputation](https://www.arxiv.org/pdf/2408.10878v3) (search summary only).
- No dead-ball evidence: Peral [excludes out-of-play sequences](https://arxiv.org/html/2512.19528), while [83% of the challenge's missing-ball frames fall in stretches predicted dead](../../docs/plan/continuar_m4_gpu.md). That split is a prediction, not a label.

### Inferences
- Plan on 3-5 m ball error in the interior of a long gap at best, and worse under degraded input. These are clean-data numbers carried across, not a measurement.
- Team accuracy climbing from 83.96% to 91.54% with only 5% of frames visible suggests anchors matter far more than the model. A gap offers anchors only at its edges, so the interior stays near the 84% level.
- Published models say little about dead-ball frames, which are our main gap population.

### Gaps
- Nobody publishes error versus distance from the nearest visible ball. A local measurement on degraded IDSSE would be new.
- No published set-piece or restart location from players alone was found. This remains an open bet (see Q4).

## Q4. Can a learned ball estimate be an extra HGB feature?

### Takeaway
Technically yes, but I found no published stacking of this kind, and the local dose-response argues against a coordinate alone reaching +0.03. Best value is a stacked **team log-odds** with ball location as an auxiliary target, plus cheap ball-proxy features. Run a one-arm noise test first.

### Cited Findings
- Local dose-response, all with degraded players ([M4 notes](../../docs/plan/continuar_m4_gpu.md), model v2, uniform gaps of real length):

  | Arm | Ball missing | Gain vs `ref41` |
  |---|---|---|
  | `noisy20` (1.5 m, 3% false) | 19% | +0.026 |
  | `noisy10` | 9% | +0.046 |
  | `ball_all` (perfect) | 0% | +0.080 |
  | `dead_m4` vs `dead41` (perfect, gaps in dead ball) | 36% vs 41% | +0.025 |

- If the 83% figure holds, only [7% of scored frames are live ball lost](../../docs/plan/continuar_m4_gpu.md).
- Those noisy arms assume [ERROR_M 1.5 m](../../possession/experiments/dose_exp.py) over recovered frames, which is better than any trajectory-only number in Q3.
- Ball-then-team loses to direct inference in one head-to-head: ["Ball TP" event F1 48.39 against 75.69](https://arxiv.org/html/2602.12080v2), per earlier note.
- Ball Radar's hierarchy predicts [possessor before ball](https://arxiv.org/abs/2306.08206), and an auxiliary possession task helps ball error: [TranSPORTmer with CLS 2.71 m, without 2.89 m](https://arxiv.org/html/2410.17785v2).
- `shape_features` in [`features_v3.py`](../../possession/features_v3.py) has x quantiles, box area, advance, final-third counts and line gap. It has no density grid, soft-min pair proxy, y quantiles or hull. Only that block was read.
- SoccerTrack v2's [soft minimum over opposing pairs](https://arxiv.org/html/2508.01802v2) needs no ball or identities.

### Inferences
Ranked for potential +0.03; all gains are unproven and I forecast no number.

1. **Dose gate on v3, before any model.** Add noise arms at 3 m and 5 m over all gaps, under uniform and dead patterns, with `mixed_all3`. Estimate: hours, CPU only. If 3 m on 100% of gaps gives under +0.03 under both, drop all coordinate estimators. Caveat: the table above is v2, so v3 may differ.
2. **Ball-proxy features in HGB.** Soft-min opposing pair distance and its location, plus density grids, are identity-free and need no new dependency. Estimate: 1 day. Risk: overlap with existing shape and control features.
3. **Cheap ball regressor as stack, no PyTorch.** Train HGB to predict ball x, y on degraded IDSSE from ball-free features, with quantiles for uncertainty; the Amirli level is the bar. Feed estimate, spread, and distance to nearest H and A. Determinism is unchanged. Estimate: 1-2 days. Honest limit: it uses information HGB already has, and adds only a richer target than the rare H/A changes.
4. **Team model with auxiliary ball head (main bet).** Frame set encoder, team pooling, Bi-LSTM or TCN at 5 fps. Outputs: H/A/D logits, ball heatmap on a coarse grid, plus team of next restart. Stack the log-odds into HGB. This extends step 3 of the earlier plan, adding the ball head, restart head and masks.
   - Train with `mixed_all3`-style degradation, contiguous gaps with lengths from the challenge's gap histogram, and dead-ball frames kept, unlike Peral.
   - Size estimate: under 0.5M params, about 1 hour per fit on the RTX 4060.
   - The restart head targets the 83%: a long dead stretch decides who owns the next spell, and the challenge has [49/59 set pieces with restart](../../BITACORA.md) detected. No paper tests whether player layout gives the restart team.
5. **FOOTPASS as no-ball training data, only if option 4 passes.** Its team-labelled events could give about 8× more matches. Blockers: CC BY-NC annotations and the [conflicting Apache-2.0 repo licence](../Posesión%20en%20tracking%20cinco%20mejoras/video_y_audio.md), possible NDA password, imputed positions, broadcast-derived domain shift, and no dead-ball labels. Ask the organisers whether NC data may be used for training.

**Leakage-safe stacking.** Out-of-fold features are mandatory, otherwise HGB trusts an over-fitted ball.
- Nested CV costs 49 fits. A cheaper scheme is leave-two-matches-out: 21 models, one per match pair.
- For outer test `t`, each training match `m` gets features from model `(m, t)`. Test features for `t` average the 6 models excluding `t`.
- The challenge match uses a model on all 7, or an average of the fold models. The second is closer to what the CV measured.
- Acceptance stays the existing gate: +0.03 paired, ≥5/7 matches, no loss on Metrica.

**Reproducibility.** Train offline, freeze weights, run in one thread on CPU with a fixed library version, and round outputs (for example to 1e-3) before HGB. A plain NumPy forward pass avoids a PyTorch dependency in delivery. Confirm in two separate processes as the project already does.

**Not worth pursuing now:** U2Diff, UniTraj, Ball Radar retraining, Event2Tracking, Peral crops, Ochin GNN.

### Gaps
- No published use of an estimated ball as a stacked feature.
- Whether residual H/A errors sit in gaps at all: the team oracle (0.909) also corrects frames with visible ball, and I found no breakdown by ball presence. `ball_all` reaching only +0.080 suggests most of the gap to 0.909 is not about missing ball. That means even a perfect estimator leaves it open.
- The real gap pattern is unknown: [`DATA.md` says the ball goes missing mainly on important frames](../../docs/plan/continuar_m4_gpu.md), and `hard41` loses 0.021. No source addresses recovery when gaps sit on possession changes.
- Borghesi et al.: a search summary says it [uses ball tracking as input](https://iris.polito.it/handle/11583/2976470), which would contradict treating it as ball-free, but full text stays blocked (403).
- Wang et al. 2014, [ball tracking via team play](https://infoscience.epfl.ch/record/185107), was seen only as an abstract, and uses video, so it is not assessed.
