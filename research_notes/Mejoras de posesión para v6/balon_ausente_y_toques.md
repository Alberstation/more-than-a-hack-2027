# Ball imputation, ball-holder inference and touch detection from player tracking (2020-2026)

Context: the ball is missing in 41% of graded frames, in gaps with a median of 10.7 s, and tracklets are anonymous. The current model is per-frame HGB plus a DeepSets/TCN encoder with Viterbi decoding, trained on 7 IDSSE/DFL (Sportec) matches and 3 Metrica games. Recall on possessions shorter than 2 s is about 20%.

Note: the "7 IDSSE matches" are the same dataset PathCRF calls the "Sportec Open DFL Dataset" (Bassek et al. 2025, 7 Bundesliga 1/2 matches). PathCRF results on that dataset can therefore be compared directly with ours, as far as data source goes.

## 1. Ball Radar, TranSPORTmer and other ball-inference/imputation models: accuracy without the ball and with a partial ball

### Takeaway
With 100% of the ball hidden, the best models put the ball about 2.6-3.7 m off and get the team in possession right 81-86% of the time. With 10-20% of the ball visible, error drops to 1.3-2.1 m and team possession accuracy rises to 95-97%. All of these models assume full, team-labelled player sets in fixed-length windows. None was tested with anonymous tracklets that come and go.

### Cited Findings
- **Ball Radar** (Kim, Choi, Kim, Yoon, Ko; KDD 2023, arXiv 2306.08206). It uses Set Transformers per team, which ignore player order, plus a hierarchical Bi-LSTM. The model first predicts who possesses the ball, then regresses the ball position. It adds a "reality loss" and postprocessing — [arXiv](https://arxiv.org/abs/2306.08206)
- Ball Radar data: 15 Fitogether GPS matches from the K League 2020 (10 train / 2 val / 3 test), which are proprietary. A Metrica-only pipeline (3 sample games: train on G1+G2, validate on the first half of G3, test on the second half) reportedly reproduces the results "with little performance loss" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206); [GitHub README](https://github.com/pientist/ballradar)
- Ball Radar inputs per player: (x, y), (vx, vy), speed and acceleration. Windows are 10 s at 10 Hz with 22 players, handled as sets per team. Only a naive-ordering ablation uses shirt numbers. Goalkeepers are often missing from GPS, so a separate GK model was trained on Metrica (5.4 m error) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Ball Radar results with no ball at all (Table 2). PE = ball position error, PPA = player-level possession accuracy, TPA = team-level possession accuracy.
  - H-LSTM-RL: PE 3.66 m, PPA 64.7%, TPA 85.85%.
  - Plain LSTM without the possessor stage: PE 5.47 m.
  - H-Transformer: PE 4.91 m, PPA 48.9%, TPA 77.9%.
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Ball Radar with part of the ball visible (Table 3, H-LSTM, before postprocessing). Training masked 80% of the ball in half of the batches — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)

| Ball hidden | PE (m) | PPA | TPA |
|---|---|---|---|
| 100% | 5.35 | 61.9% | 83.96% |
| 95% | 3.00 | 79.5% | 91.5% |
| 90% | 2.09 | 87.1% | 95.0% |
| 80% | 1.31 | 93.1% | 97.4% |

  - Postprocessing slightly hurts PPA and TPA.
- Ball Radar pass annotation (Table 4): Pass F1 was 0.39 with the ball fully hidden, 0.60 at 95% hidden, 0.74 at 90% and 0.87 at 80%. Receiver F1 was 0.57, 0.72, 0.82 and 0.90 at the same levels — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Ball Radar code: [pientist/ballradar](https://github.com/pientist/ballradar), Python 3.10 and PyTorch 1.13. The README lists no pretrained weights, and the fetched repository page showed no LICENSE file. That is consistent with "no licence", which means all rights reserved by default — [GitHub](https://github.com/pientist/ballradar)
- **TranSPORTmer** (Capellera et al., ACCV 2024, arXiv 2410.17785). One transformer handles forecasting, imputation, ball inference and state classification (pass / possession / uncontrolled / out-of-play) through a CLS token. Its set attention ignores player order — [CVF](https://openaccess.thecvf.com/content/ACCV2024/html/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.html)
- TranSPORTmer inputs are only (x, y) plus an agent type (ball / attacking player / defending player). It needs no identities and no velocities. Ball Radar, by contrast, needs velocities, ground-truth possessor labels and GK locations — [arXiv HTML](https://arxiv.org/html/2410.17785v2)
- TranSPORTmer data: 283 LaLiga 2022-23 matches, cut into 9.6 s sequences at 6.25 Hz (73,595 training sequences). With the ball 100% hidden, ADE was 2.71 m against 3.89 m for Ball Radar, and Ball Radar without its possessor stage scored 5.43 m. With 80% hidden, ADE was about 0.80-0.88 m; with 90% hidden, about 1.12-1.23 m. Four-class state accuracy with the ball hidden was 80.8%. Training ran for 100 epochs on an RTX A6000 — [arXiv HTML](https://arxiv.org/html/2410.17785v2)
- **Graph Imputer** (Omidshafiei et al., DeepMind/Liverpool FC, Scientific Reports 2022). It imputes off-screen players, not the ball, using bidirectional variational LSTMs and graph networks under simulated camera occlusion — [Sci Rep / PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9126960/); [arXiv 2106.04219](https://arxiv.org/abs/2106.04219)

### Inferences
- Our condition (41% of frames without the ball, gaps of 10.7 s) falls between the 100% and 80% rows of Table 3. Inside a gap, the model sees no ball for at least 5 s on each side. The 100%-hidden numbers (TPA 84-86%) are therefore the right reference for gap frames, not the 80% row.
- In both papers, predicting the possessor first and the ball second lowers ball error by roughly 30%: 5.47 to 3.69 m in Ball Radar, 5.43 to 3.89 m in TranSPORTmer. Possession is the easier and more useful target. For us, regressing the ball is a means, not the goal.
- TranSPORTmer's input design (x, y, team label, no identities or velocities) is the closest to our anonymous tracklets.

### Gaps
- No paper reports results for gaps of a given length (for example 10 s). Masking is random or per sequence.
- Neither model was evaluated with variable player counts or broken tracklets.
- TranSPORTmer code: I found no public repository or licence.
- Ball Radar training time and hardware are not reported.
- No physics-based or Kalman interpolation study over gaps longer than 5 s turned up in this search.

## 2. Does feeding an imputed ball or possessor distribution into a possession classifier help?

### Takeaway
The only direct comparison is PathCRF on the same Sportec/IDSSE dataset. Regressing the ball and then applying rules ("Ball TP") reaches only 48.4 event F1. Inferring the possession path directly with a masked CRF reaches 75.7 F1, with 92.4% frame-level team possession accuracy using no ball at all. The evidence favours predicting the possessor directly over passing a regressed ball downstream.

### Cited Findings
- **PathCRF** (Kim et al., arXiv 2602.12080, Feb 2026). Possession is modelled as an edge between 26 nodes (22 players plus 4 "outside" nodes), giving 676 candidate edges. Inputs are player trajectories only, with no ball. The encoder uses partially permutation-equivariant set attention (ISAB) per team, a fully equivariant ISAB over all nodes, and a temporal SAB or Bi-LSTM. A dynamic CRF masks illegal transitions with −1e4, and Viterbi decoding finds the path — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Data: Sportec Open DFL, 7 matches, split 5 train / 1 validation / 1 test. Tracking was downsampled from 25 to 5 FPS, using 10 s windows. Events were synchronised with ELASTIC, and missed touches were added with RDP plus Needleman-Wunsch alignment — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Results on the test match (Table 2):

| Method | Edge accuracy | Sender accuracy | Event F1 |
|---|---|---|---|
| Ball TP (ball regression + rules) | 47.0% | 58.6% | 48.39 (ball error 2.74 m) |
| Non-CRF | 69.2% | — | 65.34 |
| Non-CRF + valid-constrained decoding | 69.7% | — | 73.14 |
| PathCRF (dynamic masked CRF) | 69.6% | 80.4% | 75.69 (P 73.2 / R 78.4) |

  - Running at 25 FPS raises F1 to 77.21 but costs 12x more training time.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.12080)
- Downstream: frame-level team possession accuracy is 92.40%, and the home possession share comes out at 62.64% predicted against 62.13% true. Event recall is about 80% at ±1 s and ±4 m, and 86.1% at ±2 s and ±10 m — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Backbone ablation (Table 3, event F1 with the masked CRF): the Ball Radar backbone (per-team set encoder plus Bi-LSTM) gets 75.24, PathCRF's own gets 75.69, and the plain TranSPORTmer SAB gets 68.19. The decisive part is the team-aware set module, not the temporal model — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Cross-provider transfer (Appendix B): edge accuracy is about 67-68% when train and test come from the same provider. It falls to 53% for K League → Sportec and 42% for Sportec → K League — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Runtime: one RTX 4090 at 5 FPS takes 565 s per epoch for the dynamic CRF and 242 s per epoch for the static CRF. Inference for a whole match takes about 23 s. The number of epochs is not stated — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Code: [hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf) under the **MPL-2.0** licence. It includes pretrained weights (`saved/100`-`140`), the synchronised Sportec events (`data/sportec_synced/`), kloppy-based preprocessing and a tutorial notebook — [GitHub](https://github.com/hyunsungkim-ds/pathcrf)

### Inferences
- Our setup (set encoder plus Viterbi, trained on the same 7 Sportec matches) is structurally a team-level version of PathCRF. Their frame-level team accuracy of 92.4% without any ball is a concrete benchmark for our gap frames, measured on clean DFL tracking. Our target tracker is degraded, so the transfer drop (67 to 42-53% edge accuracy across providers) has to be kept in mind.
- PathCRF's real advantages over us appear to be:
  - player-level emissions (who has the ball, not just which team);
  - a learned dynamic transition that depends on the pair of players;
  - masked illegal transitions, for example a ball can go from player A only to a receiver or "out".
  - Player-level structure would directly attack the "opponent equally close, team score ~0.45" problem. The model has to commit to one player and reason about where the ball goes next, rather than averaging team proximity.
- A practical path for anonymous tracklets: keep PathCRF's per-frame player-level softmax (over the visible players of each team, masked) as an extra emission or feature. Then collapse it to team probabilities for our existing Viterbi. We would not need stable identities across the window, if the CRF stays at team level.
- Their pretrained weights assume 22 fixed nodes. Reusing them would need padding/masking and a check of how the code orders nodes. MPL-2.0 allows use, but modified files must stay MPL-2.0.

### Gaps
- PathCRF reports no results for possessions shorter than 2 s and no per-event-type numbers. Whether it fixes our short-possession recall is not measured.
- No paper tests a regressed ball fed into a learned classifier, as opposed to rules.
- The number of training epochs is not stated, so total training time on our hardware cannot be computed exactly.

## 3. Touch, reception and pass detection from tracking: accuracy and robustness to a missing or noisy ball

### Takeaway
Rule-based detectors (Vidal-Codina 2022 as implemented in databallpy, Link & Hoernig 2017) need a ball position. They detect touches from changes in ball speed or angle within about 1.5 m of a player, so they do not apply inside our ball gaps. The learned ball-free methods (PathCRF, Ball Radar) are the only ones with numbers when the ball is absent.

### Cited Findings
- **Vidal-Codina, Evans, El Fakir, Billingham** (arXiv 2202.00804, 2022). A deterministic decision tree estimates who has the ball in each frame, then combines possession changes with the laws of the game to detect events. It reports a detection rate of "+90%" in most event categories, across several tournaments and providers — [arXiv](https://arxiv.org/abs/2202.00804)
- **databallpy** implementation of Vidal-Codina:
  - A player within `pz_radius` = 1.5 m of the ball is a candidate possessor.
  - Possession is confirmed only if the ball's direction changes by more than 10° or its speed by more than 5 m/s.
  - Possession is lost when the ball leaves the 1.5 m zone.
  - Opponents both within the zone count as a "duel", which is out of scope.
  - Ball x, y and velocity are required, and the docs warn that poor ball tracking degrades results. No accuracy figures are given.
  - Source: [databallpy docs](https://databallpy.readthedocs.io/en/v0.5.0/features/team_and_player_possession.html)
- **ELASTIC** (Kim et al., MLSA@ECML-PKDD 2025, arXiv 2508.09238). It synchronises event and tracking data using only tracking-derived features, without annotated event locations, and explicitly detects the end times of pass-like events. It was evaluated on 2,134 hand-annotated events from 3 Eredivisie matches. An extended version (arXiv 2608.30227, reportedly accepted at CIKM 2026) annotates true timestamps on the Sportec Open DFL Dataset. I saw only the listing for the extended version and have not verified it — [arXiv 2508.09238](https://arxiv.org/abs/2508.09238v1); [arXiv 2608.30227](https://arxiv.org/pdf/2608.30227)
- PathCRF used ELASTIC to sync Sportec events, then added missed ball touches via RDP plus Needleman-Wunsch. Its labels therefore include short touches that the raw event feed lacks — [arXiv HTML](https://arxiv.org/html/2602.12080)

### Inferences
- Our negative result (ball-velocity touch features on a blurred ball adding nothing on top of the encoder) matches what the method predicts. The 10° / 5 m/s threshold needs a sharp ball trajectory, and blurring removes exactly that signal.
- The Sportec labels we already train on may be missing short touches. PathCRF's synced and augmented Sportec events (`data/sportec_synced/`, MPL-2.0) could be a better source of labels for spells shorter than 2 s. We would compare them against our current BallPossession labels before training.

### Gaps
- I did not retrieve numbers for Link & Hoernig 2017 (which is before 2020) or for Yamamoto 2023.
- No per-category Vidal-Codina table or temporal tolerance was retrieved, because the PDF could not be parsed.
- SoccerNet ball-action spotting was not reviewed. It is video-based and not transferable without video.

## 4. Multi-task learning: an auxiliary ball or possessor target on external data

### Takeaway
There is consistent evidence that auxiliary possession or possessor supervision improves both ball inference and event detection. I found no study that measures the reverse: an auxiliary ball-position target improving possession prediction.

### Cited Findings
- PathCRF has an auxiliary sender/receiver softmax loss on its first encoder. Removing the auxiliary losses lowers both edge accuracy and event F1 (Appendix C), down from the full model's 75.69 F1 — [arXiv HTML](https://arxiv.org/html/2602.12080)
- In Ball Radar, possessor supervision as an intermediate stage cuts ball error from 5.47 m to 3.69 m — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- In TranSPORTmer, ball error is 5.43 m without the possessor stage against 3.89 m with it. Adding a CLS state-classification head and coarse-to-fine refinement gives a further gain to 2.71 m — [arXiv HTML](https://arxiv.org/html/2410.17785v2)

### Inferences
- The most promising auxiliary target for our encoder is the **player-level possessor**: a softmax over the visible tracklets, given by the nearest player to a known ball in the Sportec/Metrica frames where the ball is present. This forces the encoder to resolve which of two nearby opponents holds the ball, which is exactly our short-spell failure.
- A ball (x, y) regression head is cheaper to add, but the papers show it as the weaker signal. In PathCRF, sending the regressed ball through rules performed poorly.
- Training with randomly masked ball (Ball Radar masks 80% of the ball in half of the batches) is a cheap way to make one model work both when the ball is present and when it is absent.

### Gaps
- No ablation was found that isolates "auxiliary ball head → possession accuracy".

## 5. What fits in under 2 days on an RTX 4060 8 GB with about 10 external matches, with deterministic CPU inference?

### Takeaway
All of the candidates are small (set encoders over about 26 nodes in 10 s windows) and were trained on 7-15 matches. The cheapest high-evidence options are: (a) a player-level possessor auxiliary head plus ball masking in our encoder, and (b) running or retraining the MPL-2.0 PathCRF code on Sportec and using its player/team probabilities as features or emissions.

### Cited Findings
- PathCRF was trained on 5 Sportec matches (about 55k windows). On a 24 GB RTX 4090 it takes 242-565 s per epoch at 5 FPS, and whole-match inference takes about 23 s. Pretrained weights are released — [arXiv HTML](https://arxiv.org/html/2602.12080); [GitHub](https://github.com/hyunsungkim-ds/pathcrf)
- Ball Radar was trained on 10 GPS matches, or on 2 Metrica games in its Metrica-only setup — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- TranSPORTmer was trained on 283 matches on an A6000, far more data than we have — [arXiv HTML](https://arxiv.org/html/2410.17785v2)

### Inferences (cost, expected gain, risk)

| Option | Cost | Expected gain | Risk |
|---|---|---|---|
| A. Possessor auxiliary head plus ball masking in our DeepSets+TCN | about 1 day | Strongest literature support (PathCRF Appendix C; the hierarchical gains above). Directly targets the 0.45 short-spell ambiguity. | Our tracklets are anonymous, so the possessor target must be defined per frame over the visible set. Gains are unmeasured on our degraded inputs. |
| B. PathCRF (MPL-2.0) retrained at team granularity, or its player softmax as features | 1.5-2 days | 92.4% frame-level team accuracy without a ball on clean DFL data, on the same matches we train with. | It assumes 22 fixed, identified nodes per window. It shows a cross-provider drop. A 4060 is roughly 2-3x slower than a 4090 (my estimate, not measured), so 5 FPS and fewer epochs are needed. |
| C. Ball Radar | about 2 days | — | No licence (all rights reserved). It needs velocities and GK positions. It is beaten by PathCRF on possession. Not recommended. |
| D. Ball regression fed downstream | — | Worst option by direct evidence (48.4 F1). Deprioritise. | — |

- Determinism on CPU: all of these are feed-forward set-attention, LSTM or TCN models with Viterbi decoding, so CPU inference in float32 with fixed seeds and `torch.use_deterministic_algorithms(True)` is feasible. That is an engineering inference; no paper reports on it.

### Gaps
- No paper reports training on an 8 GB GPU, or the total number of epochs or hours needed. The cost figures above are estimates.
