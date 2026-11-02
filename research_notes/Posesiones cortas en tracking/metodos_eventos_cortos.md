# Methods for detecting short possessions, touches, passes and receptions from tracking data

Scope note: about 20 tool calls. Primary sources were read for PathCRF, Vidal-Codina et al., TranSPORTmer, ME-DST (abstract only) and the SoccerNet 2025 report. Borghesi et al. (ESWA 2023) and Ball Radar (KDD 2023) were only reached at abstract or search-snippet level. Sanford et al. 2020, Li & Link 2026, Peral et al., NFL, handball and ice hockey were not researched (see Gaps).

## Q1. Which works detect possession, touch, pass or reception events from tracking, and what do they report on short events?

### Takeaway
The strongest directly comparable work is PathCRF (KDD 2026). It was trained and tested on the same Sportec/DFL 7 matches we use, with a per-player edge formulation and a masked CRF. It reaches event F1 75.7% at 5 fps (77.2% at 25 fps) with a 1 s tolerance, and 92.4% frame-level team-possession accuracy **without any ball data**. No source found reports accuracy broken down by possession duration. Short events are judged only indirectly, through event-level precision/recall with a time tolerance.

### Cited Findings
**PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories.** Kim, Lee, Seo, Ko, Yoon, Park. KDD 2026. arXiv 2602.12080.
- Formulation: players form a fully connected dynamic graph, and at each step exactly one edge is selected as the possession state. A socio-temporal backbone produces edge embeddings that become CRF emission and transition scores. The CRF masks impossible transitions, Viterbi decodes the path, and an event is detected whenever the selected edge changes — [arXiv abs](https://arxiv.org/abs/2602.12080)
- Data: Sportec Open DFL, 7 Bundesliga 1/2 matches (5 train, 1 val, 1 test). 25 fps is downsampled to 5 fps, with 10 s windows (50 steps) and a stride of 5 frames — [arXiv html](https://arxiv.org/html/2602.12080)
- Test-match results for Dynamic Masked CRF: edge accuracy 69.64%, sender accuracy 80.44%, receiver accuracy 82.28%, 0% violations, event precision 73.18%, recall 78.37%, F1 75.69% — [arXiv html](https://arxiv.org/html/2602.12080)
- Baselines: Non-CRF gets event F1 65.34% with 2.80% violations. Ball TP (Ball Radar heuristics) gets 48.39% F1 and a 2.74 m ball error. Static Masked CRF gets 73.83%. Dense (unmasked) CRFs get about 68% — [arXiv html](https://arxiv.org/html/2602.12080)
- Running at 25 fps raises event F1 to 77.21%, but training costs about 12x more (6727 vs 565 s/epoch) — [arXiv html](https://arxiv.org/html/2602.12080)
- Event match criterion: same type (control, kick or out-of-play), same acting player, and within 1 s. Relaxed recall is about 80% at 1 s/4 m and 86.1% at 2 s/10 m — [arXiv html](https://arxiv.org/html/2602.12080)
- Downstream: frame-level **team** possession accuracy is 92.40%, and the home possession share is 62.64% predicted vs 62.13% true — [arXiv html](https://arxiv.org/html/2602.12080)
- Loss ablation: removing the auxiliary coarse and emission losses drops Dynamic MCRF from 75.69 to 69.65 F1 — [arXiv html](https://arxiv.org/html/2602.12080)
- Cross-provider transfer is weak. Leave-one-match-out on Sportec gives F1 73.88 ± 1.90. K League → Sportec gives 66.25 ± 7.65, and Sportec → K League gives 49.16 ± 3.37 — [arXiv html](https://arxiv.org/html/2602.12080)
- No per-duration (short-possession) breakdown was found in the parts of the paper that were read — [arXiv html](https://arxiv.org/html/2602.12080)

**Automatic event detection in football using tracking data.** Vidal-Codina, Evans, El Fakir, Billingham. arXiv 2202.00804 (2022). Rule-based plus decision trees.
- Possession zone around each player has a radius of 0.5 m (Track160) or 1 m (Tracab, Hawk-Eye). The duel zone is a 1 m circle around the ball, and a duel is at least two opponents inside it — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Features, at 25 Hz: player-ball distance, ball displacement, incoming/outgoing ball direction and speed, and a Savitzky-Golay filter on ball x-y (order 2, 7 frames). A gain within a control sequence is flagged by a change in ball direction or speed above a threshold. The threshold values are in the supplementary material only — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Results: passes above 90% precision and recall. Shots 78% precision, 53% recall, with about 70% of the 254 shot errors being blocked shots labelled as passes. 569 crosses were labelled as passes. Set pieces above 90%, except free kicks — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Deflections: a blocked shot becomes a pass. A save after a deflection becomes a defender pass followed by a goalkeeper reception. Retention vs deflection is decided by whether possession is lost within 1 s. Short touches are not evaluated separately — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Data: Track160, Tracab and Hawk-Eye, from FIFA World Cups and the Bundesliga. Code is not released (Python 3 only); data is available on request from FIFA — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)

**Validation caution.** A 2024 *Big Data* study tested a rule-based pass/shot detector on four datasets. It suggests that the accuracy of positional-data event detectors is overestimated — [univie eprint](https://eprints-dev5.cs.univie.ac.at/8072) (snippet only; numbers not read).

**Borghesi, Costa, Morra, Lamberti. "Using temporal convolutional networks to estimate ball possession in soccer games."** ESWA 223 (2023), DOI 10.1016/j.eswa.2023.119780.
- Reports 86.2% possession accuracy, about 7.3 points above the state-of-the-art rule-based methods — [IRIS PoliTO](https://iris.polito.it/handle/11583/2976470)
- The earlier thesis used 2019-20 Serie A tracking and added a permutation-invariant layer over players within a team — [PoliTO thesis](https://webthesis.biblio.polito.it/18073)

**TranSPORTmer.** Capellera, Ferraz, Rubio, Agudo, Moreno-Noguer. ACCV 2024. arXiv 2410.17785.
- Uses Set Attention Blocks that alternate temporal and social attention, plus a CLS "extra agent" that classifies each frame's state as pass, possession, uncontrolled or out-of-play — [arXiv html](https://arxiv.org/html/2410.17785v2)
- State accuracy is about 87-90% with ball observed (Table 1) and about 80.84% with the ball fully masked. "Uncontrolled" is the weakest class, and no per-class F1 is reported. Data: LaLiga 2022-23, 283 matches, at 6.25 Hz — [arXiv html](https://arxiv.org/html/2410.17785v2)
- No code link — [arXiv html](https://arxiv.org/html/2410.17785v2)

**SoccerNet Team Ball Action Spotting 2025 (video, not tracking).**
- 12 classes, including Pass, Drive, Header, High Pass and Ball Player Block. The winner reached Team-mAP@1 60.03, against a 51.72 baseline built on T-DEED. The gain came from a unified joint action-team head — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)

**Player-centric ball action spotting (SoccerNet 2026, FOOTPASS dataset).**
- ME-DST (Wang, Yang, Wang, arXiv 2608.01696) keeps a per-player role-slot axis, with temporal attention per slot and spatial attention across slots. It adds role embeddings, tracking-derived tactical features and fused X3D-L/Swin3D-S video predictions. It reports Micro-F1 0.778, 10.3 points above the TAAD+DST baseline. Ablations say preserving the entity axis and role identity drive the gain — [arXiv abs](https://arxiv.org/abs/2608.01696)
- Other entries: per-player attention in DST gives Macro-F1 58.94 ([arXiv 2606.28389](https://arxiv.org/abs/2606.28389)). A GNN-DST fusion with square-root class weights moves Macro-F1 from 0.493 to 0.505 ([arXiv 2606.09679](https://arxiv.org/pdf/2606.09679)). Micro and Macro F1 are not comparable — [SoccerNet 2026 results](https://arxiv.org/pdf/2607.07320)

**Basketball.**
- Older work used CRFs to predict ball ownership from player positions only. No recent SportVU pass/touch detector with numbers was found in this pass — [search summary of IEEE Access / Sloan sources](https://sloansportsconference.com/research-papers/using-deep-learning-to-understand-patterns-of-player-movement-in-the-nba)

### Inferences
- PathCRF's ball-free team accuracy of 92.4% on DFL is a useful reference point. However, it is measured on clean (non-degraded) Sportec tracking, and all frames are weighted equally.
- In PathCRF, event F1 rises by about 10 points when the CRF masks illegal transitions. Our constant-switch-cost 3-state Viterbi has no comparable player-level structure.

### Gaps
- No paper found reports recall by possession duration (<2 s etc.). PathCRF's 1 s event tolerance is the closest proxy.
- The Borghesi TCN details (receptive field, classes, short-possession error analysis) were not reached because the PDF returned 403.
- Not researched: Sanford et al. 2020, Peral et al., Li & Link 2026, NFL Big Data Bowl, handball, ice hockey.

## Q2. Which input representations help short events?

### Takeaway
The measured evidence favours three things: per-player (edge or slot) representations, structured decoding that forbids impossible transitions, and keeping the entity axis rather than pooling. Higher frame rate gives a modest gain (+1.5 F1, 5 to 25 fps, PathCRF). Ball-kinematics rules (speed/direction change) work for passes when the ball is clean.

### Cited Findings
- PathCRF backbone ablation: replacing the SAB social module (TranSPORTmer-style) with PPE-FPE (Ball Radar-style player/team encoders) moves edge accuracy from 62.97% to 69.64% and event F1 from 68.19 to 75.69. The temporal module choice (Bi-LSTM vs PE-SAB) matters only marginally (75.24 vs 75.69) — [arXiv html](https://arxiv.org/html/2602.12080)
- Frame rate: going from 5 to 25 fps takes event F1 from 75.69 to 77.21 in PathCRF — [arXiv html](https://arxiv.org/html/2602.12080)
- Rule-based gains are triggered by ball direction or speed change after Savitzky-Golay smoothing at 25 Hz, inside a 0.5-1 m player zone — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- ME-DST attributes its gain to keeping the per-player entity axis and role embeddings — [arXiv abs](https://arxiv.org/abs/2608.01696)
- The 2025 SoccerNet winner gained from a joint action-team head over a dual head — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)

### Inferences
- Our DeepSets mean/max pooling per team discards which player is near the ball. Both PathCRF and ME-DST report losses when this entity axis is collapsed. This points to player-level emissions (who is in possession) as the likely next step, rather than more team-level features. This has not been measured on our data.

### Gaps
- No source isolates the effect of ball velocity/acceleration features under 40% ball missingness.

## Q3. Do player-level holder models beat team-level classifiers on short possessions?

### Takeaway
There is indirect evidence only. A player-level path model with a masked CRF gets 92.4% team-frame accuracy without the ball, and ME-DST ablations favour the entity axis. No paper compares player- and team-level models by possession length.

### Cited Findings
- PathCRF gets 92.40% team-frame accuracy as a by-product of player-edge inference, ball-free — [arXiv html](https://arxiv.org/html/2602.12080)
- Ball Radar has "direct" and "team" model variants (Section 4.2 of the paper), but the numbers were not read — [GitHub ballradar](https://github.com/pientist/ballradar)
- ME-DST: preserving the entity axis is "central" to its +10.3 Micro-F1 — [arXiv abs](https://arxiv.org/abs/2608.01696)

### Inferences
- Since PathCRF uses the same 7 DFL matches, it could be run on our degraded version and scored with our duration-bucket recall to get the missing comparison directly.

### Gaps
- The Ball Radar player-level vs team-level numbers were not reached (the arXiv 2306.08206 abstract has no numbers).

## Q4. Reusable repositories and licences

### Takeaway
PathCRF is the best candidate: MPL-2.0, DFL data with synced events included, and checkpoints provided. Ball Radar has public code but no licence was found.

### Cited Findings
- **PathCRF**, github.com/hyunsungkim-ds/pathcrf. MPL-2.0. Includes train/inference/evaluate scripts, a tutorial notebook, `data/sportec_synced` (events synced with ELASTIC) and checkpoints for all 5 variants (`saved/100`-`140`) — [GitHub](https://github.com/hyunsungkim-ds/pathcrf)
- **Ball Radar**, github.com/pientist/ballradar (KDD 2023). Set Transformer plus hierarchical Bi-LSTM, with a possessor classifier and a ball regressor. Proprietary Fitogether data; reproducible on 3 Metrica matches. PyTorch 1.13. No licence shown — [GitHub](https://github.com/pientist/ballradar)
- **Vidal-Codina et al.**: no code released — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00804)
- **TranSPORTmer**: no code link in the paper — [arXiv html](https://arxiv.org/html/2410.17785v2)
- **ME-DST**: no code link on the abstract page — [arXiv abs](https://arxiv.org/abs/2608.01696)

### Inferences
- MPL-2.0 is file-level copyleft. Modified PathCRF files must stay MPL, but they can be combined with our own code.

### Gaps
- Licences for the Borghesi TCN and the SoccerNet 2026 entries were not checked.
