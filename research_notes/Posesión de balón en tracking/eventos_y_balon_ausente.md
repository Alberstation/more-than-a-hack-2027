# Detecting on-ball events from tracking data, and inferring the ball when it is missing (state as of Oct 2026)

Context: hackathon "More than a Hack 2027". The task is to label every frame H/A/D using automatic, uncorrected 25 fps tracking. Players have a team but no identity, tracklets are fragmented, and the ball is missing in about 41% of frames. A possession change counts only when the other team gains controlled possession. The baseline (nearest player + majority vote) scores spellF1 0.2433.

## Prioritized shortlist (summary for the report writer)

| # | Resource | Type | Code / license | Key reported numbers | Fit for the hackathon |
|---|---|---|---|---|---|
| 1 | **PathCRF** (Kim et al., KDD'26, arXiv 2602.12080) | DL, ball-free: possession path from players only | [github.com/hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf), MPL-2.0, weights included | Event P 73.2 / R 78.4 / F1 75.7%; sender acc 80.4%, receiver 82.3% | Very high conceptually (fits the 41% missing ball). Needs retraining or adaptation to anonymous tracklets. |
| 2 | **Ball Radar** (Kim et al., KDD'23, arXiv 2306.08206) | DL: ball trajectory + possessor from players; also imputation | [github.com/pientist/ballradar](https://github.com/pientist/ballradar), no license file | PE 3.66 m with no ball; 1.31 m with 20% of the ball observed; PPA 64.7%, TPA 85.9% | High. Already processes **Metrica** data (same provider). Good for imputation. |
| 3 | **ELASTIC** (Kim et al., 2025 + CIKM'26, arXiv 2508.09238 / 2608.30227) | Rule-based / scoring event-tracking synchronization with receive detection | [github.com/hyunsungkim-ds/elastic](https://github.com/hyunsungkim-ds/elastic), MPL-2.0 | 88.4% exact frame (vs ETSY 21.1%); receive 92.6% exact | Needs event data, so it does not apply directly to the target match. Useful for labelling Sportec training data and for reusing its touch features. |
| 4 | **Vidal-Codina et al. 2022** (Sports Engineering, arXiv 2202.00804) | Rule-based possession + events | No public code; FIFA data private; paper CC BY 4.0 | Pass R 94 / P 90%; shot R 53 / P 78%; set pieces >90% | High as a recipe (gain = change in direction/speed inside a PZ). Assumes good ball tracking. |
| 5 | **Bischofberger et al. 2024** (PLOS ONE) | Rule-based hit/play detection (ball acceleration + proximity) | No repo found | Metrica: pass F 0.89, shot 0.65; independent Euro data: pass 0.71 | High: simple, calibrated **on Metrica** (min_acc 26.8 m/s², vicinity 1.4 m, deflection 91 ms). |
| 6 | **TranSPORTmer** (Capellera et al., arXiv 2410.17785) | Transformer: ball inference/imputation + per-frame state (pass/possession/uncontrolled/out) | No code found; paper CC BY-NC-SA | Ball inference ADE 2.71 m (vs Ball Radar 3.89 m); state acc ~80.8% | Medium: the idea of a "controlled vs uncontrolled" state fits the task, but there is no code. |
| 7 | **ETSY** (Van Roy, Cascioli, Davis, MLSA'23) | Rule-based sync | [github.com/ML-KULeuven/ETSY](https://github.com/ML-KULeuven/ETSY), Apache-2.0 | 21.1% exact / 74.7% within ±1 s (as measured by ELASTIC) | Low-medium. Needs SPADL events. |
| 8 | **databallpy** | Python package: loading + sync (cost function) | [github.com/alek050/databallpy](https://github.com/alek050/databallpy), JOSS | No public metric (the docs say there is no ground truth) | Medium: loaders for Metrica/Tracab/Sportec. |
| 9 | **Link & Hoernig 2017** (PLOS ONE) | Individual/team possession (IBP/TBP) with a Bayesian network for degree of control | Data on figshare | Earlier version: P/R 80–92% on clean data | Medium: the conceptual definition of "controlled possession" matches the hackathon rule. |
| 10 | **Sportec Open DFL Dataset** (Bassek et al. 2025, Sci Data) | Labelled data: 7 matches, 25 fps tracking + official events | CC BY 4.0, kloppy / HF IDSSE | n/a | Key as **training data** from other matches (the rules allow it). |

---

## Q1. Rule-based event detectors from tracking (Vidal-Codina, Bischofberger, Link & Hoernig, ETSY, ELASTIC, databallpy, DFL/Sportec)

### Takeaway
Rule-based detectors reach about 90% on passes **when the ball tracking is good**: Vidal-Codina gets R 94 / P 90%, and Bischofberger gets F 0.89 on Metrica. With event data independent of the positional data, the pass F-score drops to about 0.71. All of them depend on the ball, so with 41% of frames missing they need imputation (Q3) or a ball-free method (Q2). Synchronizers (ETSY, ELASTIC, databallpy) require an event feed, so they do not run on the target match. Their touch-detection features (ball acceleration, player-ball distance, kick distance) can still be reused.

### Cited Findings
**Vidal-Codina, Evans, El Fakir, Billingham (2022), "Automatic event detection in football using tracking data", Sports Engineering; arXiv 2202.00804 (CC BY 4.0)**
- Two deterministic steps, decision-tree style. (1) A possession model: who controls the ball in each frame, using a "possession zone" (PZ) and a "duel zone". (2) Events derived from possession changes plus the laws of the game: passes, shots, crosses, saves, receptions, interceptions, set pieces. — [arXiv abs](https://arxiv.org/abs/2202.00804v3)
- Loss rule: the ball leaves player A's PZ with displacement above ϵs, **and** A is not in the next control frame with possession or a duel. This means a player moving away from a static ball, or a player who loses and regains the ball with no one else touching it, is **not** a loss. — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- Gain rule: within a control sequence inside one player's PZ, a gain is counted only if the ball changes **direction** (d0·d1 < ϵθ) **or speed** (|Δv| > ϵv). If the ball passes through unaltered, there is no gain. This proxies real contact when no z coordinate is available. — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- PZ radius was 0.5 m for Track160 and 1 m for Tracab/Hawk-Eye; duel zone 1 m. Data: 25 Hz, 2D ball only. Tested on FIFA Club World Cup 2019/2020 and Bundesliga 2019-20 matches (Track160, Tracab, Hawk-Eye), benchmarked against official STS/Sportec events (>30K passing events). — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- Open play results: pass recall 94% / precision 90%; shot recall 53% / precision 78% (blocked shots get labelled as passes); cross recall 39% / precision 86%. Kick-offs and penalties 100%; corners, throw-ins and goal kicks >90%. Runs in about 3 min per match on 1 CPU (Python 3). — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- Stated limitations: it **needs ball tracking** and an in-play/dead-ball flag. Player swaps hurt it. The authors suggest enlarging the PZ and retuning hyperparameters for lower-quality data (single camera or broadcast). — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- Data private (request via research@fifa.org); no public code found. — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3); [search](https://dspace.mit.edu/handle/1721.1/145347)

**Bischofberger, Baca, Schikuta (2024), "Event detection in football: Improving the reliability of match analysis", PLOS ONE 19(4)**
- Rule: a "hit" is ball acceleration > min_acc while the nearest player is within min_vicinity. A "play" is two consecutive hits by different players, or a hit followed by an interruption. A deflection filter discards plays shorter than max_deflection_time. Shots are separated by 5 rules, or by AdaBoost. — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/)
- Parameters fitted **on Metrica** (3 matches, 25 Hz): min_acc 26.8 m/s², min_vicinity 1.4 m, max_deflection_time 91 ms. These vary widely by dataset (Euro: 63.3 m/s², 1.5 m, 99 ms; Stats: 25.9, 3.0, 748). — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/)
- Play detection on Metrica: P 0.89, pass recall 0.90, shot recall 0.70, F 0.83. Full pipeline on Metrica: pass F′ 0.89, shot 0.65. On Euro (events independent of tracking, the most realistic set): pass F′ 0.71, shot 0.62. — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/)
- The authors argue that reported accuracies in the literature are **overestimated** because events and tracking are not independent. — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/); [univie eprints](https://eprints.cs.univie.ac.at/8072)
- Python 3.9 with numpy/pandas/scikit-learn; no repo URL in the article. — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/)

**Link & Hoernig (2017), "Individual ball possession in soccer", PLOS ONE (DOI 10.1371/journal.pone.0179953)**
- Defines team ball possession (TBP) as an unbroken sequence of individual possession (IBP) phases. A Bayesian network classifies the degree of control from the configuration of the player, the ball and opponents. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225); [figshare data](https://figshare.com/articles/dataset/Individual_ball_possession_in_soccer/5190445)
- The 2016 version (Hoernig, Link et al., "Detection of individual ball possession in soccer") reports precision/recall of 80–92% on "flawless" data. — [TUM portal](https://portal.fis.tum.de/en/publications/detection-of-individual-ball-possession-in-soccer/) (as summarized by search; not verified in full text)

**ETSY (Van Roy, Cascioli, Davis, ECML/PKDD MLSA 2023), KU Leuven DTAI**
- Rule-based sync of events (SPADL format, socceraction) to tracking. It corrects the kick-off offset per period, then matches each event within a window using a distance-based score. Apache-2.0, Python ≥3.10, 4 commits. — [GitHub ETSY](https://github.com/ML-KULeuven/ETSY); [DTAI blog](https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization)

**ELASTIC (Kim, Choi, Lee et al., KAIST + AFC Ajax; arXiv 2508.09238, Aug 2025; extended "Trajectory-Based Synchronization..." arXiv 2608.30227, Aug 31, 2026, accepted at CIKM 2026 according to a co-author's LinkedIn)**
- Does not use annotated event locations, only tracking features. Events are grouped as pass-like, set-piece, incoming and minor. Candidate frames are scored on player-ball distance, ball acceleration, kick distance after/before the event, and frame delay. Includes **receive detection** (end frame and receiver of each pass, or out/goal). Minor events (tackle, foul, bad touch, take-on, dispossessed) use specific rules. — [arXiv 2508.09238 html](https://arxiv.org/html/2508.09238v1)
- On 2,134 events (3 Eredivisie 2024-25 first halves, re-annotated at 25 fps by 3 annotators, 90.4% exact agreement): start exact **88.4%** vs ETSY 21.1%; within 25 frames 95.0% vs 74.7%; mean error 2.35 vs 15.15 frames. Pass-like exact 95.5% (ETSY 21.9%); set pieces 92.3%; incoming 85.7%; **minor events only 44.4%**; receive exact 92.6%. — [arXiv 2508.09238 html](https://arxiv.org/html/2508.09238v1)
- The 2026 version adds virtual termination events plus Needleman-Wunsch alignment, and a public benchmark from re-annotated Sportec matches. The abstract gives no numbers. — [arXiv 2608.30227](https://arxiv.org/abs/2608.30227)
- Repo: MPL-2.0, Python ≥3.9, Sportec tutorial, `evaluate.py` comparing etsy/biermann/databallpy/elastic_greedy/elastic_nw. — [GitHub elastic](https://github.com/hyunsungkim-ds/elastic)
- ELASTIC cites Biermann et al. at <10% exact alignment. — [arXiv 2508.09238 html](https://arxiv.org/html/2508.09238v1)

**databallpy (Oonk et al., JOSS)**
- Loads Tracab/Sportec/Metrica and syncs events to tracking. Approaches: by timestamp, or with a cost function. The docs admit there is no "perfect" dataset to compare methods against. — [databallpy docs sync](https://databallpy.readthedocs.io/en/main/getting_started/synchronisation_page.html); [GitHub](https://github.com/alek050/databallpy); [JOSS](https://joss.theoj.org/papers/10.21105/joss.10223)

**DFL/Sportec synchronized data (Bassek, Rein, Weber, Memmert 2025, Scientific Data 12:195)**
- 7 matches (Bundesliga 1 and 2, 2022/23), 25 fps tracking with ball, plus official synchronized events. CC BY 4.0. Mirror on HF "pysport/idsse-data" with kloppy loaders. — [PMC11787359](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/); [HF README](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)

### Inferences
- For the hackathon rule ("touch, deflection or aerial duel without control does not count"), the Vidal-Codina logic is directly useful: a gain needs a change of direction/speed inside the PZ, and a loss is not counted if the same player recovers. Combine it with the organisers' persistence proxy (keeps the ball a few seconds, or >2 consecutive passes) as a filter on top of detected gains.
- Bischofberger's Metrica-calibrated thresholds (26.8 m/s², 1.4 m, 91 ms) are a reasonable starting point if the target match is also Metrica tracking. The automatic, uncorrected tracking is noisier than the sample data, so expect to retune min_vicinity upward (Vidal-Codina also suggests a larger PZ for lower-quality data).
- The synchronizers (ETSY/ELASTIC/databallpy) do not apply to the target match (no event data there). They are valuable for **generating per-frame labels** (possessor, pass start/end) on Sportec/Metrica matches and training a model, which the rules allow. PathCRF already ships ELASTIC-synced Sportec events.
- Minor events (tackles/duels) are hard even with event data (44% exact in ELASTIC). Do not count on detecting tackles explicitly. It is better to infer them as "the ball ends up controlled by the other team".

### Gaps
- No public code for Vidal-Codina found (no repo linked on arXiv or the article page). The online supplementary material (S3–S4 hyperparameters) was not reviewed.
- Morgulev et al. (cited in the brief) was not researched. No specific open repo found for detecting duels/tackles from tracking.
- No public precision/recall for databallpy on Sportec.
- ELASTIC CIKM 2026 numbers are not available in the abstract.

## Q2. ML/DL event and possession detection from tracking (including ball-free)

### Takeaway
The most relevant 2026 line of work is **PathCRF** (KAIST, KDD'26). From player trajectories alone (no ball), it infers a "possession path" (control vs kick between players) with a masked CRF plus Viterbi decoding. It reaches event F1 75.7% on a Sportec test match and beats the "regress the ball, then apply heuristics" approach (F1 48.4%). It comes with code, weights and the Sportec pipeline under MPL-2.0. TranSPORTmer and Peral et al. (WACV'26) classify per-frame ball state / possessor, but no code was found.

### Cited Findings
- **PathCRF** (Kim, Lee, Seo, Park [KAIST], Ko [U. Seoul], Yoon [Fitogether]; arXiv 2602.12080, Feb 2026, KDD'26): a fully connected dynamic graph of 22 players + 4 "outside" nodes. At each step it selects one edge: a self-loop means control, sender→receiver means a kick. Socio-temporal backbone (Set-Transformer-style PPE/FPE + temporal SAB). A dynamic masked CRF forbids impossible transitions, and Viterbi decodes. An event is detected when the edge changes. — [arXiv html](https://arxiv.org/html/2602.12080v1)
- PathCRF data: Sportec Open DFL (5 train / 1 val / 1 test matches), downsampled to 5 fps, in-play only. Best (Dynamic Masked CRF): edge acc 69.6%, sender 80.4%, receiver 82.3%, 0% violations, **event P 73.2 / R 78.4 / F1 75.7%**. Baselines: Ball TP (Ball Radar + heuristics) F1 48.4%; non-CRF 65.3%; non-CRF + Viterbi 73.1%; Static Masked CRF 73.8%. — [arXiv html](https://arxiv.org/html/2602.12080v1)
- PathCRF repo: MPL-2.0, 15 commits, pretrained weights in `saved/100–140`, `datatools/preprocess.py` produces tracking with per-frame possession states, `tutorial.ipynb`, ELASTIC-synced Sportec events included, uses kloppy. — [GitHub pathcrf](https://github.com/hyunsungkim-ds/pathcrf)
- **TranSPORTmer** (Capellera et al., arXiv 2410.17785, Nov 2024): a single transformer for forecasting, imputation and ball inference. A CLS token classifies the per-frame state: **pass / possession / uncontrolled / out-of-play**, with about 80.8% accuracy during ball inference. LaLiga 2022-23 data, 283 matches, 6.25 Hz. No code URL; paper CC BY-NC-SA 4.0. — [arXiv html v2](https://arxiv.org/html/2410.17785v2)
- **Peral, Capellera et al., "Multi-Modal Soccer Scene Analysis with Masked Pre-Training"** (WACV 2026, arXiv 2512.19528): ball trajectory inference, ball state and possessor from trajectories, player type and **player image crops** (tactical camera). CropDrop pre-training. The abstract gives no metrics and no code. — [arXiv abs](https://arxiv.org/abs/2512.19528)
- Video-based (not tracking) action spotting as a reference: FOOTPASS / SoccerNet 2026 player-centric ball-action spotting reaches Macro F1 0.548 on test. — [arXiv 2606.09679](https://arxiv.org/pdf/2606.09679); [FOOTPASS arXiv 2511.16183](https://arxiv.org/pdf/2511.16183)

### Inferences
- PathCRF formulates exactly the hackathon problem: possession per frame without relying on the ball, with structural constraints that prevent absurd flips. The CRF/Viterbi idea can be reused even without the deep network. For example, run an HMM/Viterbi over the states {H, A, D}, or {possessor_i}, with emissions from proximity/velocity features and a transition penalty. That directly reduces short-segment fragmentation, which is what spellF1 penalizes.
- PathCRF assumes 22 consistent player slots. With anonymous, fragmented tracklets you would need to (a) work per team (H/A + "D"), with permutation-invariant emissions per frame, or (b) stitch tracklets first. This is an inference: the paper does not evaluate noisy tracking.
- The domain gap is Sportec (Tracab-like, high quality) to Metrica automatic tracking. Expect degradation. Training/fine-tuning on Metrica sample data plus Sportec is feasible in 3 weeks on a GPU.

### Gaps
- No public evaluation of PathCRF or TranSPORTmer on noisy automatic tracking or with anonymous IDs.
- No code found for TranSPORTmer or Peral et al.
- TacticAI-style GNN work does not target event detection; it was not researched further.

## Q3. Ball trajectory imputation/inference when the ball is missing

### Takeaway
**Ball Radar** (KDD'23) is the most practical option. Its public code already processes Metrica sample data. From players alone it gives a mean error of 3.66 m and 85.9% team-level possession accuracy. With partial observations, the error drops sharply (1.31 m and 93% PPA with 20% of the ball visible), which is close to the case of a ball missing 41% of the time. TranSPORTmer reports better errors (2.71 m with no ball; about 0.84 m with 80% masked) but has no public code.

### Cited Findings
- **Ball Radar** (Kim et al., KDD 2023, arXiv 2306.08206, DOI 10.1145/3580305.3599779). Hierarchy: a Ball Possessor Classifier (Set Transformer + Bi-LSTM), then a Ball Trajectory Regressor conditioned on possession probabilities. Adds a "reality loss" and physical post-processing. Suggested applications include imputation and semi-automatic pass annotation. — [arXiv abs](https://arxiv.org/abs/2306.08206)
- Data: Fitogether GPS (K League, 15 matches, private) + **Metrica sample** (3 matches). 10 s windows at 10 Hz. Results (Table 2): H-LSTM-RL PE **3.66 m**, PPA 64.7%, **TPA 85.85%**. Transformer-based variants are worse (PE 4.7–5.4 m, TPA ~78%); VRNN 12.37 m. — [arXiv html](https://arxiv.org/html/2306.08206v1)
- Imputation (Table 3, H-LSTM-RL), as PE / PPA before post-processing: 100% masked 5.35 m / 61.9%; 95% masked 3.00 m / 79.5%; 90% masked 2.09 m / 87.1%; **80% masked 1.31 m / 93.1%**. — [arXiv html](https://arxiv.org/html/2306.08206v1)
- Caveats: no variance across seeds; small test set; reassigned player IDs in one Metrica match. — [arXiv html](https://arxiv.org/html/2306.08206v1)
- Repo: Python 3.10, PyTorch 1.13, Metrica processing notebook, `model_eval.ipynb`, TraceAnimator. **No LICENSE file**, no pretrained weights mentioned. — [GitHub ballradar](https://github.com/pientist/ballradar)
- **TranSPORTmer** on soccer: ball inference ADE 2.71 m / MaxErr 7.39 m vs Ball Radar 3.89 / 8.79 (on their data). Imputation ADE about 0.84 m (80% masked) and 1.09 m (90% masked). The fetch notes the table row alignment is unclear, so verify against the PDF. — [arXiv html v2](https://arxiv.org/html/2410.17785v2)
- **MIDAS** (KETI/Kim et al., arXiv 2408.10878): multi-agent imputation (Set Transformer + Bi-LSTM) for **players**, not the ball. Per a secondary summary, it beats linear/spline/BRITS/NAOMI/ImputeFormer (not verified). — [arXiv html](https://arxiv.org/html/2408.10878v4)

### Inferences
- Practical strategy: train Ball Radar (or its possession classifier alone) on Metrica sample + Sportec. Apply it with the observed ball as a partial input (the 60% visible), and use the inferred team possession (TPA ~86% without a ball, much higher with partial ball) as an emission in an HMM/Viterbi. PathCRF shows that "regress the ball, then apply heuristics" is worse than predicting possession directly, so prioritize the possession output over the ball trajectory.
- Ball Radar's player-level possession (PPA 65%) needs player identity. Team-level possession (TPA) is what matters for H/A/D and is more robust to anonymous IDs.

### Gaps
- No public metric for imputing the ball over long gaps (>several seconds) in automatic tracking.
- The NRTSI / "hierarchical ball trajectory" works mentioned in the brief were not verified. No soccer-specific ball results with code were found beyond Ball Radar.

## Q4. What is proven and practical for ~3 weeks in Python?

### Takeaway
Practical ranking: (1) a rule-based Vidal-Codina/Bischofberger-style detector (gain = direction/speed change near a player; Metrica thresholds), plus persistence filters, plus HMM/Viterbi smoothing. This takes 1 week and has no training risk. (2) Ball Radar for filling ball gaps and estimating team possession, trained on Metrica sample + Sportec (code works with Metrica). (3) Adapting PathCRF (code + weights + Sportec) if there is GPU time, as the strongest ball-free model. ELASTIC/ETSY/databallpy are only used to build training labels on public matches.

### Cited Findings
- Vidal-Codina: Python, about 3 min per match on 1 CPU. — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- Bischofberger: thresholds already fitted for Metrica; numpy/pandas/sklearn only. — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/)
- Ball Radar: Metrica pipeline ready, PyTorch 1.13. — [GitHub ballradar](https://github.com/pientist/ballradar)
- PathCRF: weights + tutorial + ELASTIC-synced Sportec, MPL-2.0. — [GitHub pathcrf](https://github.com/hyunsungkim-ds/pathcrf)
- Sportec Open DFL: 7 labelled matches, CC BY 4.0, kloppy loaders. — [PMC11787359](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/); [HF](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- Warning on overestimated accuracy when evaluating with dependent data. — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/)

### Inferences
- License: ballradar has no license file (legally "all rights reserved"). Using it in a hackathon is likely tolerated, but redistributing code derived from it is risky. PathCRF/ELASTIC are MPL-2.0 (file-level copyleft) and ETSY is Apache-2.0, so all are fine.
- Validation without hand labels on the target match: tune on Metrica sample matches (official events) and Sportec, measuring spellF1 with possession segments derived from those events.
- Everything runs locally, so the no-cloud-upload constraint is satisfied.

### Gaps
- No published method evaluates a "controlled possession" definition equivalent to the hackathon's (persistence ≥ a few seconds / >2 passes). It has to be implemented as a post-filter.
- None of the methods has been validated on automatic tracking without identities and with heavy fragmentation.
