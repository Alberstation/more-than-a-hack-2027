# Domain adaptation and robustness: clean optical tracking to automatic / broadcast tracking (possession and event models)

Scope note: about 15 searches and fetches. "Verified" means the number was read on the fetched page or search snippet. Items marked "(not verified in this session)" come from well-known papers whose specific numbers were not checked here. Leave them out or check them before citing.

## Q1. Measured accuracy drop of tracking-based models across providers or from clean to broadcast tracking

### Takeaway
PathCRF (Kim et al., 2026) gives the only directly comparable possession/event number found here. Moving from one provider to another (DFL camera-based ↔ K League GPS) cuts event F1 by 8 to 31 points and edge (possessor) accuracy by 14 to 27 points. The authors attribute the drop to provider noise and annotation conventions. Validity studies of tracking itself show optical systems at about 0.08 m RMSE and broadcast/AI systems at 1.7 to 16 m RMSE. Provider identity matters a lot.

### Cited Findings
- **PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories.** Hyunsung Kim et al. (KAIST and others), arXiv 2602.12080, Feb 2026. It infers the possessor as a path on a dynamic player graph with a masked CRF and Viterbi decoding, and needs no ball. On the Sportec/IDSSE DFL test match it reports edge acc 69.64%, event P/R/F1 73.18/78.37/75.69%, frame-level team-possession accuracy 92.40%, and home possession share 62.64% predicted vs 62.13% true. Training used 5 train / 1 val / 1 test match (379 episodes, 54,683 windows; test has 1,831 events). Code: https://github.com/hyunsungkim-ds/pathcrf (code licence not stated in the paper; the paper itself is CC BY 4.0). — [arXiv HTML](https://arxiv.org/html/2602.12080)
- **PathCRF Appendix B, cross-provider results (Table 6, mean ± SD).** The second dataset is proprietary K League 1: 15 matches of 10 Hz GPS, split 10/2/3. Sportec used leave-one-match-out over 7 matches. — [arXiv HTML](https://arxiv.org/html/2602.12080)

  | Train → Test | Edge acc | Event P | Event R | Event F1 |
  |---|---|---|---|---|
  | K League → K League | 68.20 ± 2.64 | 84.34 ± 2.42 | 75.79 ± 2.76 | 79.83 ± 2.60 |
  | Sportec → Sportec | 66.75 ± 2.76 | 72.05 ± 2.37 | 75.87 ± 2.33 | 73.88 ± 1.90 |
  | K League → Sportec | 52.71 ± 9.31 | 71.93 ± 6.43 | 61.52 ± 8.41 | 66.25 ± 7.65 |
  | Sportec → K League | 41.65 ± 2.34 | 54.44 ± 3.68 | 44.84 ± 3.28 | 49.16 ± 3.37 |

  - In both directions, recall falls more than precision (Sportec→KL: R −31 pts, P −30 pts; KL→Sportec: R −14 pts, P −0.1 pt). Under shift the model misses possession changes more often than it invents them. That matches our under-segmentation symptom.
  - Possession accuracy and noise-robustness experiments are not reported in the appendix. The authors call the drop "provider distribution shift" and argue that users would train on their own data.
- **Linke, Link & Lames (2020). "Football-specific validity of TRACAB's optical video tracking systems."** PLOS ONE 15(3): e0230179. Setup: 20 players, VICON reference, sport-specific running course plus small-sided games. Position RMSE was 0.08 m (Gen5) and 0.09 m (Gen4). Speed and acceleration differences were statistically significant but trivial. Gen4 underestimated low-speed distance by about 19%. Peak speed, acceleration and deceleration differed by less than 0.5%. Both systems are "valid … as long as players are tracked correctly". — [PMC7064167](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064167/)
- **Crang, Johnston, Mills, Billingham, Robertson, Cole, Weakley, Hewitt, Duthie (2025). "Concurrent validity of computer-vision artificial intelligence player tracking software using broadcast footage."** arXiv 2508.19477. Setup: one FIFA WC2022 match, three unnamed commercial broadcast-tracking providers, TRACAB Gen5 reference, tested on tactical, programme and camera-1 feeds. Position RMSE was 1.68 to 16.39 m and speed RMSE 0.34 to 2.38 m/s. Total-distance bias ranged from −21.8% to +24.3%. Provider had a significant effect. The tactical feed maximises detection. 720p and 1080p were both adequate. Accuracy is "fair" only when players are detected. — [arXiv](https://arxiv.org/abs/2508.19477)
- **"Broadcast analytics – an evaluation of video-based tracking systems with constrained player visibility" (2025, SPONET record).** About 97% of player activity is visible in the scouting feed and less than 50% in the TV feed. Tactical metrics such as formation recognition held up, with minor drops on TV. Physical metrics were more sensitive to missing data. The authors call for transparency about provider interpolation. — [SPONET](https://www.sponet.de/sponet/Record/4096118?lng=en)
- **SkillCorner vendor validation (not peer-reviewed).** Setup: 10 camera positions at the Stadium of Light, compared against official EPL tracking. Player identification was about 90 to 98% and ball tracking above 92% for most camera positions. — [SkillCorner article](https://skillcorner.com/articles/does-camera-position-and-capture-style-affect-tracking-quality)
- **Vidal-Codina, Evans, El Fakir, Billingham (2022). "Automatic event detection in football using tracking data."** Sports Engineering. A rule-based possession model plus event detector, applied to Track160, Tracab and Hawk-Eye data. Detection rate is above 90% for most event categories across tournaments and providers (per-category numbers not retrieved). — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)

### Inferences
- PathCRF's cross-provider drop (event F1 −7.6 to −30.7 pts) is the best available prior for how far a model trained on DFL-only data moves on another provider. Our setup has more shift on top of that: no identities, 40.7% ball missing, fragmented tracks.
- Most of PathCRF's loss is in recall. For us the matching move is to recalibrate the change-point threshold or segment rate on the target, rather than retraining the whole model (see Q2).
- The two-stage, rule-informed design of Vidal-Codina held above 90% across three providers, so explicit physical rules transfer reasonably. A hybrid of rules and a learned model is worth testing as a robustness baseline.

### Gaps
- No per-provider numbers for the three broadcast providers in Crang et al. were retrieved. SkillCorner is not named.
- No published study was found that measures possession-segmentation rate (possessions per minute) under clean-to-broadcast shift.
- No independent peer-reviewed SkillCorner-vs-optical study was found. Taberner et al. and Ellens et al. were not reached in this session's budget.

## Q2. Noise simulation, test-time adaptation, self-training, prior-shift correction, segment-rate calibration

### Takeaway
Imputation work (MIDAS/DBHP, Graph Imputer, ball-trajectory inference) gives realistic sports-specific missingness models with code: random gaps, whole-agent dropout, and camera-view occlusion. No sports paper was found that uses tracker-error augmentation to make a possession model robust. For adapting to the unlabelled target, the general methods are well established:
- BN-statistic adaptation and Tent (entropy minimisation on norm-layer affine parameters)
- SHOT (information maximisation plus pseudo-labels)
- EM prior correction with bias-corrected calibration (Alexandari et al. 2020), which TableShift's finding supports, since label shift correlates with OOD gap at ρ ≈ 0.7

### Cited Findings
- **MIDAS / DBHP. Choi et al., "Trajectory Imputation in Multi-Agent Sports Using Derivative-Based Hybrid Prediction"** (arXiv 2408.10878; MIDAS version at ECML/PKDD 2025). A permutation-equivariant Set Transformer with derivative-accumulating self-ensemble. It is tested under three missingness regimes: uniform random gaps, agent-wise dropout, and broadcast-camera occlusion. It is also evaluated on downstream tasks (total distance, pass success). — [arXiv](https://arxiv.org/abs/2408.10878v4)
- **Graph Imputer** (Omidshafiei et al., Liverpool FC / DeepMind, Scientific Reports 2022). Predicts off-screen players when only on-screen players are observed, which is a camera-visibility missingness model. — [Nature Sci Rep](https://www.nature.com/articles/s41598-022-12547-0.pdf)
- **Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM** (Kim et al., KDD 2023, arXiv 2306.08206). Infers the ball from player trajectories and masks the true ball randomly in training. It is directly relevant to our 40.7% ball-missing case. — [arXiv](https://arxiv.org/pdf/2306.08206)
- **TacSIm (CVPR 2026 supplement).** Gaps under 5 frames get motion-constrained splines and longer gaps get a conditional VAE imputer. It notes that coarser spatial grids are less sensitive to noise. — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wen_TacSIm_A_Dataset_CVPR_2026_supplemental.pdf)
- **Kontos & Karlis (2023).** Compare interpolation methods for player and ball coordinates censored by the broadcast camera. — cited in [search summary of fupress IJAS 35(2)](https://oajournals.fupress.net/index.php/ijas/article/view/15707)
- **Tent: Fully Test-time Adaptation by Entropy Minimization.** Wang, Shelhamer, Liu, Olshausen, Darrell, ICLR 2021. Online, it re-estimates normalisation statistics and optimises channel-wise affine parameters to minimise prediction entropy, with no source data and no labels. It set state of the art on ImageNet-C at the time. Specific error numbers were not verified here. Code: github.com/DequanWang/tent (not verified in this session). — [arXiv 2006.10726](https://arxiv.org/abs/2006.10726)
- **BN-statistic adaptation (Schneider et al., NeurIPS 2020, "Improving robustness against common corruptions by covariate shift adaptation")** and **SHOT (Liang, Hu, Feng, ICML 2020, source-hypothesis transfer: freeze the classifier, adapt the feature extractor with information maximisation plus self-supervised pseudo-labels)** are standard source-free baselines. (Neither was verified in this session. arXiv 2006.16971 and 2002.08546.)
- **Alexandari, Kundaje, Shrikumar (ICML 2020). "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation."** Method: calibrate on held-out source data, then run Saerens-style EM on target predictions to re-estimate class priors. The likelihood is concave, so EM reaches the global optimum, with no retraining and no hyperparameters. It beats BBSL and RLLS on MNIST, CIFAR-10/100 and diabetic retinopathy. Bias-corrected calibration (class-wise biases) beats plain temperature scaling. Code: https://github.com/kundajelab/labelshiftexperiments. — [arXiv 1901.06852](https://arxiv.org/abs/1901.06852v5)
- **TableShift (Gardner, Popovic, Schmidt, NeurIPS 2023 D&B).** Across 15 tabular shift tasks, the shift gap correlates with label-distribution change Δy (ρ = 0.71). ID accuracy plus Δy explains OOD accuracy with R² ≈ 0.99. Label-shift robustness methods did not close the gaps and often lowered both ID and OOD accuracy. Domain-robustness methods narrowed the gap mainly by lowering ID accuracy. — [arXiv 2312.07577](https://arxiv.org/html/2312.07577v3)

### Inferences
- Our symptom is that confidence on the target is unchanged (0.944 vs 0.94) while the segment rate is low (3.2 vs 4.23 per live minute). That fits a change-point detector whose boundary scores are compressed under noise, since smoothing and fragmentation blur ball-carrier switches. It does not fit a model that has become uncertain. Entropy minimisation (Tent) would make the model more confident and is unlikely to add boundaries on its own.
- Better-targeted options:
  1. **Rate/prior correction.** Treat "boundary at frame t" as the class with a shifted prior. Run EM (Alexandari) on the target's boundary probabilities, or set the decoding threshold or HMM/CRF transition penalty so the predicted rate on unlabelled target minutes matches a target rate. The target rate could be the IDSSE ratio (0.89×) applied to a plausible LaLiga rate. The 5-minute public ground truth (4.23/min) can serve only as validation.
  2. **Self-training.** Pseudo-label high-confidence segments on the full target match, retrain or fine-tune the GBDT with the target rows weighted, and check against the 5-minute ground truth.
  3. **Augmentation realism.** Add the MIDAS camera-occlusion and agent-wise regimes and ID-switch/fragment simulation to the existing ball-gap, dropout, noise and dead-ball simulators.
- Following TableShift, the OOD gap is largely predictable from label shift. Measuring the class/boundary prior on the 5-minute window gives most of the information needed to correct.

### Gaps
- No sports paper was found that applies Tent, SHOT, or pseudo-labelling to tracking-based possession or event models.
- No empirical tracker-error model was found for automatic tactical-camera systems (ID-switch rates, fragment-length distributions). The Crang et al. RMSE ranges are the closest.
- No literature was found specifically on calibrating segment count under shift. Change-point and HMM penalty tuning is a standard approach, but no football-specific source was found.

## Q3. Broadcast tracking datasets with possession or events, and whether training on them helps transfer

### Takeaway
Three usable public broadcast or automatic datasets with events exist:
- **SkillCorner Open Data:** 10 A-League 2024/25 matches with dynamic events and phases of play
- **PFF FC WC2022:** 64 matches, broadcast tracking plus events
- **SoccerTrack v2:** 10 panoramic 4K matches with 12-class ball actions

SoccerNet-GSR shows how hard automatic reconstruction from broadcast is. No paper was found that measures whether training on these datasets improves transfer to a different automatic-tracking source. PathCRF's table suggests that training on the target-like provider is what recovers performance.

### Cited Findings
- **SkillCorner Open Data** (with PySport). Broadcast tracking for 10 A-League 2024/25 matches plus derived Dynamic Events, phases of play and season aggregates. Event x/y are not scaled to standard pitch size. Licence not confirmed in this session, so check the LICENSE in the repo. — [GitHub](https://www.github.com/SkillCorner/opendata), [docs](https://skillcorner.github.io/opendata/)
- **PFF FC 2022 World Cup dataset.** Free release of tracking and event data. One source says it covers all 64 matches, with events quality-checked against video and tracking. Broadcast tracking is collected via Sportlogiq. PFF FC is now Gradient Sports. kloppy has a PFF tracking loader. — [PFF blog](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data), [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data)
- **SoccerTrack v2** (Scott et al., Nagoya / Tsukuba, arXiv 2508.01802, Aug 2025). 10 full university matches, 932 min of fixed panoramic 4K BePro video. Per-frame pitch coordinates, jersey IDs, roles and teams. 12 ball-action classes (Pass, Drive, Shot, Header, Cross, …) from BePro events, manually corrected. GS-HOTA degrades with sequence length. No BAS numbers or dataset licence in v1. Code: https://github.com/AtomScott/SoccerTrack-v2; data: huggingface.co/datasets/atomscott/soccertrack-v2. — [arXiv HTML](https://arxiv.org/html/2508.01802v1), [project](https://atomscott.github.io/SoccerTrack-v2/)
- **SoccerNet Game State Reconstruction** (Somers et al., CVPRW 2024, arXiv 2404.11335). Minimap reconstruction from broadcast, scored with GS-HOTA (HOTA plus role, team and jersey). The baseline scored 22.26% on test in the paper and 23.36 on the 2024 leaderboard (DetA 9.80, AssA 55.69). The 2024 winner (Constructor Tech) scored 63.81, ahead of UPCxMobius at 43.15 and JAM at 34.40. Code: https://github.com/SoccerNet/sn-gamestate (TrackLab). — [GSR paper](https://arxiv.org/pdf/2404.11335), [2024 results](https://arxiv.org/pdf/2409.10587), [SOTA method](https://arxiv.org/pdf/2504.06357)

### Inferences
- Running our pipeline on SoccerNet-GSR outputs, or on the SoccerTrack v2 pipeline, would generate labelled automatic-tracking data with realistic fragmentation. Because these come with ground-truth identities and events, they are the closest public stand-in for our LaLiga input. The gap between the baseline (23 GS-HOTA) and the winner (64) also provides two noise levels for a robustness curve.
- SkillCorner A-League adds broadcast-style missingness plus provider events for multi-source training. Event definitions differ from DFL's, which PathCRF identifies as part of its cross-provider drop, so labels need harmonising (for example, deriving possessions from passes and receptions the same way for every source).

### Gaps
- No published experiment was found that trains on broadcast datasets and measures the gain on a third automatic-tracking source.
- Exact licences for the SkillCorner open data and PFF WC2022 were not confirmed.

## Q4. GBDT vs neural models under distribution shift

### Takeaway
The best evidence comes from tabular benchmarks, not sports. In TableShift, no neural or robustness method consistently beats XGBoost, LightGBM or CatBoost in-distribution or out of distribution, and OOD accuracy follows ID accuracy linearly. No sports-tracking study comparing GBDT and neural transfer was found.

### Cited Findings
- TableShift: "no model consistently beats" the tree baselines on ID performance or robustness. ID and OOD accuracy follow a strong linear trend. Robustness methods only narrow gaps at the cost of ID accuracy. — [arXiv 2312.07577](https://arxiv.org/html/2312.07577v3)
- Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022), "Why do tree-based models still outperform deep learning on tabular data?" (arXiv 2207.08815). Trees are more robust to uninformative features and irregular target functions (not verified in this session).
- PathCRF (a neural CRF over a player graph) drops 8 to 31 F1 points across providers. There is no GBDT comparison for that shift. — [arXiv HTML](https://arxiv.org/html/2602.12080)

### Inferences
- With an engineered-feature HGB, the choice of features probably matters more for transfer than GBDT vs neural. Features that are invariant to identity, count and fragmentation are the ones to aim for: nearest-opponent distances, ball-relative order statistics, team-level aggregates.
- For a set encoder, the target-side tools are norm-statistic adaptation and Tent. For GBDT, prior/threshold correction and pseudo-label retraining are the equivalents.
- An ablation on the existing simulators would settle the question for our data: HGB vs set encoder evaluated at each degradation level.

### Gaps
- Per-model OOD numbers from TableShift (Tables 14 and 15) were not retrieved.
- No sports-specific GBDT vs neural transfer study was found.
- Taberner et al. and Ellens et al. validity studies were not retrieved within this session's budget.
