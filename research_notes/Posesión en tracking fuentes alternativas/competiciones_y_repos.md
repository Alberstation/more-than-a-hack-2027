# Competition solutions and repos with components transferable to possession from degraded tracking (state at 2026-11-01)

Scope: new sources beyond `research_notes/Posesión en tracking cinco mejoras/`. PathCRF, Ball Radar, TranSPORTmer, Peral, Borghesi and Baouan are already covered there and are only cross-referenced here. No downloaded code was run. Repo licences, last push dates and star counts come from the public GitHub API on 2026-11-01. A licence of "None" means the API detected no licence, so default copyright applies: reuse the idea, not the code.

Source quality: most competition facts below are from READMEs or official challenge reports. Several paper facts were read through a summarising fetch tool, not line by line. Items resting on search summaries or aggregators are flagged.

## Objective: ranked list of reusable components

### Takeaway
No top-placed solution found solves H vs A without the ball. The best transferable items are structural: legal-restart anchors for the decoder, aggregation over a segment instead of per-frame decisions, symmetric test-time averaging, and a second-stage GBDT over context. Space and shape features are cheap, but none has published evidence on H vs A for anonymous tracklets.

### Cited Findings

| # | Component | Type | Source and evidence | Mapping to this task | Licence | Effort, risk |
|---|---|---|---|---|---|---|
| 1 | Law-based restart triggers | Feature, decoder evidence | [Vidal-Codina et al.](https://ar5iv.labs.arxiv.org/html/2202.00804): corners, throw-ins, goal kicks above 90% precision and recall, with ball and clean feed | Detect a restart at D to live, then bias the team on exit. See Q4 | Paper; no code release stated | Low. Tolerances need retuning on noise |
| 2 | Vote over the segment | Post-processing | [SoccerNet 2025](https://arxiv.org/pdf/2508.19182): GSR winner KIST-GSR stabilises team labels by majority vote within tracklets (search summary). [Li & Link](https://arxiv.org/html/2606.09289v1): sequence modelling was the main driver of gains | Pool P(H given live) over each candidate spell before the team decision, rather than leaving only frame sums in Viterbi | Concept | Low |
| 3 | Flip test-time averaging | Augmentation, inference | [lRomul](https://github.com/lRomul/ball-action-spotting), SoccerNet BAS 2023 first place; BAS-6 in [SoccerNet 2024](https://arxiv.org/html/2409.10587v1) | Average pitch-flip and home-away-swap versions. Our report says `predict_v2` does not do the swap average | MIT | Very low. Deterministic |
| 4 | Gaussian smoothing of scores | Post-processing | [lRomul](https://github.com/lRomul/ball-action-spotting): sigma 3.0, then peak picking with height 0.2, distance 15 frames | Smooth emissions or a change-point score before decoding. Values are video-specific, do not copy them | MIT | Very low |
| 5 | Out-of-fold hard-negative sampling | Training | [lRomul](https://github.com/lRomul/ball-action-spotting): sampling distribution merged with OOF predictions by element-wise max | Oversample live frames where OOF team confidence is wrong, mainly near changes | MIT | Low |
| 6 | Filter, deep model, GBDT stack | Architecture | [nvnnghia](https://github.com/nvnnghia/1st_place_kaggle_player_contact_detection), described as NFL contact first place: XGBoost filter, 3D CNN, XGBoost post-processing. Another top team used shift and aggregation features in stage 2, per [a slide summary](https://speakerdeck.com/sugupoko/yin-202302-kaggle-nfl-player-contact-detection) | Stage 2 HGB fed with OOF probabilities of stage 1 at lags and window aggregates, for wider context in ambiguous stretches | No licence detected | Low. Needs nested OOF |
| 7 | Hungarian template fit | Feature | [unravelsports EFPI](https://unravelsports.readthedocs.io/en/stable/api/generated/unravel.soccer.EFPI.html): optimal assignment to scaled templates, attack and defence detected separately; `shaw-glickman` templates available | Fit each team to in-possession and out-of-possession templates learned on IDSSE. Feature is the cost difference. See Q2 | MPL-2.0 | Medium. EFPI drops frames with under 10 outfield players per team |
| 8 | Time-to-intercept matrix | Feature | [Bekkers, Pressing Intensity](https://arxiv.org/abs/2501.04712): Spearman intercept time with velocity and reaction, logistic map; code in [unravelsports](https://github.com/UnravelSports/unravelsports) | Compute in both directions; use the asymmetry as a who-is-pressed signal, with target as all opponents. Untested without a ball | MPL-2.0 | Medium. Velocity noise |
| 9 | Team Voronoi share | Feature | [floodlight](https://floodlight.readthedocs.io/en/latest/modules/models/models.html) DiscreteVoronoiModel; [databallpy](https://databallpy.readthedocs.io/en/latest/features/space_occupation.html) uses a KDTree over a grid | Share by zone (own half, final third), with sign by attack direction | MIT | Low |
| 10 | Delaunay cross-team edges | Feature | [Renner, Koster, Bauer, Schienle](https://www.emergentmind.com/papers/2610.11571): best GNN-LSTM setup uses a Delaunay graph approximating passing lanes, on 203 matches (aggregator page) | Hand-built counts of mixed vs same-team edges and same-team edge length per team | Paper | Low. Claimed gain includes possession input |
| 11 | Bird's-eye rendering | Input | First-place NFL contact approach rendered tracking as images, per [slides](https://speakerdeck.com/k951286/kaggle-nfl3konpezhen-rifan-ri) | A small CNN over team occupancy grids. Baouan's grid reached 80.03% per [our report](../../reports/Posesión%20en%20tracking%20cinco%20mejoras.md) | Concept | Medium. Neural CPU determinism needs its own proof |
| 12 | Player-as-token transformer | Architecture | [SportsTrackingTransformer](https://github.com/SumerSports/SportsTrackingTransformer): 4.61 vs 5.78 yards ADE against fixed-grid baseline; a 52K-parameter version reached 4.95. [Powered the 2025 BDB winner](https://sumersports.com/the-zone/sumersports-big-data-bowl-2026) | Identity-free set encoder as emission model. Only after the PathCRF ceiling test | No licence detected | High |
| 13 | Boundary-tolerant labels | Training | [lRomul](https://github.com/lRomul/ball-action-spotting): positives in a 0.6 s window; temporal jitter of ±1 frame at 40% | Jitter change points by label-noise scale in augmentation | MIT | Low |
| 14 | Weighted focal loss | Loss | [Li & Link](https://arxiv.org/html/2606.09289v1); BAS-6 alpha 0.4, gamma 1.2 in [SoccerNet 2024](https://arxiv.org/html/2409.10587v1) | Our report set focal aside because evidence is mixed; these do not overturn that | Concept | Low priority |

### Inferences
- Test order: 3, 2, 1, 6, then 7 to 10 as a single feature batch. Items 11 and 12 add neural risk. This ordering is my judgement, not measured.
- Items 2 and 6 target the decision directly, which matches the finding that the gap is about the decision rather than the data. Clean data reaches only 0.614 against the oracle's 0.909.

### Gaps
- No published effect size exists for any of these on H vs A from degraded tracking.

## Q1. Winning solutions: SoccerNet, DFL, NFL, NBA, hockey

### Takeaway
Most winners are video-first and solve a different problem. Their decoding and training patterns transfer, their models do not. No football, NBA or hockey competition with possession from tracking without a ball was found.

### Cited Findings
**SoccerNet**
- BAS 2023 first place: lRomul, 86.47% mAP@1 on challenge. Slow fusion gave about 15 points over 2D early fusion, per the author. 2D EfficientNetV2 B0, then 3D blocks, GeM pooling — [repo](https://github.com/lRomul/ball-action-spotting).
- BAS 2024 winner: T-DEED, 73.39 mAP@1, trained jointly on two datasets with two heads, then a two-model ensemble. Second place Habel et al. 71.35, three models with different validation holdouts. BAS-5 used reciprocal square root of class frequency as sampling weight — [SoccerNet 2024](https://arxiv.org/html/2409.10587v1).
- GSR 2024 winner: Constructor.tech, 63.81. Team model on 111 uniforms; tracklets joined on times, ReID, team, jersey and player speed; DeepSORT run in pitch coordinates. GSR-16 clustered embeddings with KMeans and scored below baseline — [SoccerNet 2024](https://arxiv.org/html/2409.10587v1).
- GSR 2025 winner: KIST-GSR 63.90 over Constructor.Tech 63.81, with tracklet majority voting for team labels — [SoccerNet 2025](https://arxiv.org/pdf/2508.19182) (search summary, not read in full).

**NFL**
- Contact detection: XGBoost filter, CNN, XGBoost post-processing, separate player and ground models — [repo](https://github.com/nvnnghia/1st_place_kaggle_player_contact_detection). Winner attribution conflicts: the repo claims first place, while [a walkthrough digest](https://recapio.com/digest/kaggle-winners-walkthroughs-1st-and-future-player-contact-detection-with-team-mtss-by-kaggle) names team MTSS. These may be the same team; not confirmed.
- Tracking was rendered as bird's-eye images with target players as large circles, not used as tabular features — [slides](https://speakerdeck.com/k951286/kaggle-nfl3konpezhen-rifan-ri) (secondary).
- Big Data Bowl 2025 winners Sandwar and Bajaj used Sumer's transformer notebook — [Sumer](https://sumersports.com/the-zone/sumersports-big-data-bowl-2026). It treats all 22 players simultaneously with self-attention — [repo](https://github.com/SumerSports/SportsTrackingTransformer).

**DFL Bundesliga Data Shootout**
- 500+ teams; a DFL representative said the winning solution was less complex than expected — [DFL](https://www.dfl.de/en/innovation/bundesliga-data-shootout-machine-learning-competition-on-football-data/). lRomul says his approach was very inspired by the top solutions — [repo](https://github.com/lRomul/ball-action-spotting).

**NBA and hockey**
- Wei et al. (ICCV 2015 workshop) predict the ball owner from player trajectories with role-ordered features, and note a ball detector can be fused in — [record](https://lida.sport-iat.de/dbb/Record/4039908?lng=en).
- A patent describes awarding possession during instants when the puck is not seen, using a zone of influence and minimum time — [patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9025021). It is a patent, not an evaluation.
- Tora, Chen and Little (CVPRW 2017) use an RNN on frame and player-appearance features for puck possession events — [record](https://lida.sport-iat.de/twm/Record/4078292?lng=en). This is video, not tracking.
- Waterloo's PLUCC infers puck location from gaze and body position in broadcast video — [Waterloo](https://uwaterloo.ca/artificial-intelligence-institute/news/how-ai-can-track-hockey-action-faceoff-finish).

### Inferences
- Role ordering from Wei is the older form of the identity problem. Sorted per-team quantiles already proposed in our report are the same idea, and need no fixed slots.
- The persistence rule in the hockey patent corresponds to our switch cost. It adds nothing new.
- T-DEED's joint training on a related dataset suggests a second head on a related label, such as event-based control, could regularise the H vs A head. Weak analogy.

### Gaps
- The DFL top write-ups were not reachable by search, so no specific technique from them is reported. Direct reading of the Kaggle discussion tab is needed.
- SoccerNet Tracking winners and the NFL Big Data Bowl 2026 Prediction first place were not found.
- I found no accuracy figure for Wei or Tora's possession task.

## Q2. Phase of play and team shape

### Takeaway
Reported possession from shape alone stays near 80-82% (Baouan, already in our notes). Recent high phase scores use possession as input, so they do not measure the H vs A task. Width, length and hull separate attacking from defending in small studies, but with reported exceptions.

### Cited Findings
- Li & Link: T-GAN on 7 Bundesliga TRACAB matches at 25 Hz; macro F1 0.87 intentions, 0.76 invasion, 0.79 scoring; IoT-D F1 rises 0.68 to 0.79 with post-processing. Possession is an input — [arXiv](https://arxiv.org/html/2606.09289v1). Attention falls mostly on wingers, then midfielders, and the authors warn against reading this as influence.
- The same paper reports Bauer et al. 2023 at F1 0.45 for attacking play and Biermann et al. 2025 at about 80% accuracy for binary GNN phases — [arXiv](https://arxiv.org/html/2606.09289v1). Second-hand; originals not read.
- Renner et al.: second-by-second seven-class GNN-LSTM on 203 matches with rule-based labels; beats the best baseline by 4.6% macro F1; uses possession and ball-status indicators — [aggregator](https://www.emergentmind.com/papers/2610.11571).
- FIFA EFI classifies frames into in- and out-of-possession phases from spatial and physical features, with no published accuracy found — [FIFA](https://fifatrainingcentre.com/media/native/tournaments/womens-world-cup/2023/FIFA%20Enhanced%20Football%20Intelligence%20(EFI)%20Explainations_EN%20v1.1.pdf).
- Shaw & Glickman measure in- and out-of-possession formations separately on 180 matches, from average vectors between neighbouring players — [paper](https://www.sportperformanceanalysis.com/s/Dynamic-analysis-of-team-strategy-in-professional-football-By-Laurie-Shaw-And-Mark-Glickman.pdf). That distinction is the premise of component 7.
- Bialkowski et al. measure positions relative to team mean and assign roles to minimise entropy — [paper](https://la.disneyresearch.com/wp-content/uploads/Identifying-Team-Style-in-Soccer-using-Formations-from-Spatiotemporal-Tracking-Data-Paper.pdf). A 2025 extension uses soft role assignment — [arXiv](https://arxiv.org/html/2502.03342v1).
- 8-match tracking study: attacking teams had larger surface area, stretch index, length and width — [Lisbon repository](https://repositorio.ulisboa.pt/bitstreams/d1267a66-fa5d-43ef-88fa-acd6a65407f6/download) (search summary).
- Bundesliga, 153 games: almost no whole-team compactness difference between successful and unsuccessful defensive plays — [Vienna](https://ucrisportal.univie.ac.at/en/publications/is-a-compact-organization-important-for-defensive-success-in-elit/) (search summary). This compares defensive outcomes, not possession state.

### Inferences
- A useful feature is relative: hull or width of H minus A, and last-line height relative to halfway. Absolute values vary with team style, which matters with one challenge match.
- Soft role assignment fits fragmented tracklets better than Hungarian on a full XI, because missing players do not force a hard mismatch.
- Shape changes slowly, so it can settle long ambiguous stretches, but not locate short spells or boundaries. This agrees with our report.
- Shape labels from IDSSE may not generalise to the challenge's teams. A per-match relative feature limits that risk but does not remove it.

### Gaps
- No common benchmark of H vs A accuracy from shape on open data was found.
- I could not isolate which single shape signal is best: no per-feature ablation was found. Li & Link also report none.

## Q3. Pitch control and space without ball or identity

### Takeaway
Add Voronoi-style space by zone first. It needs only positions and team side, and MIT implementations exist. Spearman-style models depend on the ball, so only the time-to-reach core is useful here.

### Cited Findings
| Tool | What it offers | Licence, last push |
|---|---|---|
| [floodlight](https://floodlight.readthedocs.io/en/latest/modules/models/models.html) | Centroid, stretch index, nearest mate and opponent, team spread, convex hull, discrete Voronoi, kinematics | MIT, 2026-05-04 |
| [databallpy](https://github.com/Alek050/databallpy) | Voronoi and Gaussian occupation, pressure (Herold & Kempe), player possession (Vidal-Codina), DAS, event sync | MIT, 2026-10-29 |
| [unravelsports](https://github.com/UnravelSports/unravelsports) | Pressing intensity, EFPI, graph conversion, kloppy-based loading | MPL-2.0, 2026-01-16 |
| [kloppy](https://github.com/PySport/kloppy) | Provider loading | BSD-3-Clause, 2026-10-30 |
| [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) | Formation templates used by EFPI | MIT, 2026-10-27 |
| [socceraction](https://github.com/ML-KULeuven/socceraction) | Event-based valuation; not tracking | MIT, 2026-10-18 |
| [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) | Spearman implementation used in comparisons | MIT, 2022-11-03 |
| [accessible-space](https://pypi.org/project/accessible-space/) | Per-frame AS and DAS through pass simulation | MIT, [repo](https://github.com/jonas-bischofberger/accessible-space) 2026-10-30, 5 stars |
| [KNNPitchControl](https://github.com/nvsclub/KNNPitchControl) | Neighbour-based ownership | None detected |
| [Metrica-pitch-control](https://github.com/anenglishgoat/Metrica-pitch-control) | Fernandez and Spearman in PyTorch | AGPL-3.0: avoid |

- The KNN paper states such models "cannot be valued quantitatively", and compares them only visually on 20 goal sequences; KNN uses positions, velocities and team label — [arXiv](https://ar5iv.labs.arxiv.org/html/2501.05870). So no accuracy ranking of these models exists there.
- Spearman's formula depends on time to reach a location, implicitly the ball's — [same paper](https://ar5iv.labs.arxiv.org/html/2501.05870). Thomson's implementation is said to run a match in under 30 seconds (same source).
- The databallpy docs note that Voronoi ignores player motion — [docs](https://databallpy.readthedocs.io/en/latest/features/space_occupation.html).

### Inferences
- Priority: Voronoi share by zone, then the intercept asymmetry (component 8), then KNN. Skip DAS unless ball-free use is shown, since it models passes from the ball.
- Velocity-weighted variants may be harmed by noisy tracking. Compare plain Voronoi first.
- Reimplement in NumPy for hash stability instead of adding a dependency. Grid and tie-break must be fixed.
- From memory, not checked here: Fernandez and Bornn scale player influence with distance to the ball, which would make the Gaussian model ball-dependent as published. Verify before use.
- Voronoi share may largely echo the team centroid, which v2 already has, so its marginal value could be small.

### Gaps
- No published use of any of these as a possession feature without the ball was found.
- No checked evidence on `accessible-space` working without ball position.
- No licence check on the DeepSport repos relevant to this task. They are basketball vision challenges and appear marginal; only the [re-identification repo](https://github.com/DeepSportradar/player-reidentification-challenge) was checked (Apache-2.0).

## Q4. Restart detection and post-restart priors

### Takeaway
Restarts are the most promising new team evidence: player shape identifies the type and often the executing team. The published rules need adapting, because they use the ball and a dead-ball flag.

### Cited Findings
- Vidal-Codina's triggers: kick-off, all on own side with one near the centre mark; penalty, keeper on line, one opponent at the mark, others outside; goal kick, a player in own goal area; corner, a player near a corner mark; throw-in, a player beyond the sideline; free kick, default fallback — [paper](https://ar5iv.labs.arxiv.org/html/2202.00804). Numeric tolerances are in Online Resource 1, not read.
- Confirmation needs the ball in the player's zone on the first in-play frame, and dead ball is from a provider flag or missing ball — [paper](https://ar5iv.labs.arxiv.org/html/2202.00804). Free kicks are the weakest class.
- Data for that paper are not public — [paper](https://ar5iv.labs.arxiv.org/html/2202.00804).
- Learned set-piece classification shows promise on corners, free kicks and throw-ins, but the author calls it far from live use — [MIT thesis](https://dspace.mit.edu/handle/1721.1/156805).
- Throw-ins, Premier League 2018-19, 16,154 throws: 83% successful first contact, 54% possession retained, 8.8% led to a shot, according to a later paper citing it — [Stone, Smith & Barry](https://shura.shu.ac.uk/27964/); [follow-up](https://shura.shu.ac.uk/35132/).
- Five leagues 2022/23: retention 92.13% backwards, 71.20% lateral, 49.75% forwards — [follow-up](https://shura.shu.ac.uk/35132/). The two papers' rates use different datasets, so do not merge them.

### Inferences
- Replace ball confirmation with a shape-only rule: a player from team T beyond the touchline during predicted D, with opponents retreating, gives log-odds for T at restart. By law the thrower's team restarts, so the first spell's team prior is very high at that moment, then decays. 54% is a longer-horizon retention and should not be used as the emission at the restart.
- Throw direction cannot be observed before release, so the 92/71/50 split is useful for simulation, not as a prior.
- For goal kick, the team is the keeper's. For a corner, it is the attacking team, whose side is known from period direction. For a kick-off after a goal, the team is the side that conceded, and this also makes goal detection valuable.
- Fit priors on IDSSE, with labels and events, not from these papers. Compare with Smart Tagging, with 24 transitions of each type, which are far fewer than the number of restarts.
- Noise can create false beyond-line detections. Require persistence over a short interval and use only frames in a predicted D state. Risk: a player off the pitch for treatment or a substitution.

### Gaps
- No open per-restart-type accuracy without the ball was found.
- Restart counts per match and recall needed under gaps should come from IDSSE, not literature.
- I did not find retention by restart type on a common definition for goal kicks, corners or kick-offs.

## Q5. Private or gated code

### Takeaway
I found no statement of any private repo available on request. Routes below are public channels. Authors' emails were not verified, so use the paper's correspondence field. The relevant ask is usually for code or thresholds, not data.

### Cited Findings
- Vidal-Codina et al.: data on request to research@fifa.org with a research proposal; code release not stated — [paper](https://ar5iv.labs.arxiv.org/html/2202.00804). Numeric tolerances are likely already in its [supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs12283-022-00381-6/MediaObjects/12283_2022_381_MOESM1_ESM.pdf), so read it before writing.
- Li & Link: labels annotated by the first author; no code or data availability found in the text read — [arXiv](https://arxiv.org/html/2606.09289v1).
- Renner, Koster, Bauer, Schienle: no code link on the aggregator — [page](https://www.emergentmind.com/papers/2610.11571).
- Sumer: public code, no licence detected — [repo](https://github.com/SumerSports/SportsTrackingTransformer). A GitHub issue asking for a licence is the proper route.
- nvnnghia: public code, no licence detected — [repo](https://github.com/nvnnghia/1st_place_kaggle_player_contact_detection).
- Baouan et al.: no GitHub link in the PDF, per [our earlier note](../Posesión%20en%20tracking%20cinco%20mejoras/equipo_sin_balon.md). Ball Radar's repo lacks a root licence, per the same note; an issue to Hyunsung Kim is the route.

### Inferences
- Best value per request: (1) Baouan for embedding code and split details, (2) Sumer and Ball Radar for a licence, (3) Li & Link for Appendix features and whether a no-possession variant exists. Vidal-Codina's data is unlikely to pay, since IDSSE covers the same use.
- Check challenge rules before contacting anyone or using gated data. No challenge data may be shared in any request.

### Gaps
- Authors' reply policies are unknown. I did not confirm any contact email other than the FIFA address.
