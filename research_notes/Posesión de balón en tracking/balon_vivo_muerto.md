# Ball-in-play vs dead-ball detection (tracking + video) for football

Research date: 2026-10-28. Scope: per-frame alive/dead classification at 25 fps, restart detection (throw-in, corner, goal kick, free kick, kick-off, penalty), stoppage start/end timing. Target setting: one LaLiga match, uncorrected automatic tracking (ball missing ~41% of frames), Smart Tagging set-piece intervals, homography, tactical wide-angle video.

## 1. Open-source methods / heuristics for ball-in-play from tracking, and their accuracy

### Takeaway
There are two directly relevant published methods. Lang et al. (2022, Sci. Rep.) classifies in-play vs stoppage from player positions only: about 92% frame accuracy and about 1.7-1.8 s mean boundary error, with no public code. Vidal-Codina et al. (2022, Sports Eng.) gives deterministic, IFAB-based player-configuration triggers for restarts with more than 90% precision and recall. It assumes the alive/dead flag is given, though: provider boolean, or ball missing when dead. That assumption is exactly the one our baseline gets wrong. Neither has an off-the-shelf repo, but both are simple enough to reimplement in a few days.

### Cited Findings
**Lang, Wild, Isenko, Link (2022), "Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data", Scientific Reports. CC BY 4.0 article, no public code, data not public.** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/) / [Springer](https://link.springer.com/10.1038/s41598-022-19948-1)
- Data: 102 Bundesliga / 2. Bundesliga 2017/18 matches with TRACAB optical tracking at 25 fps. In-game status was annotated by humans. Split: 45 train, 10 validation, 47 test. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- Features use players only, no ball. For all 22 players: x, y, velocity and acceleration per frame, plus time-shifted copies at ±0.4, ±0.8, ±2, ±4, ±20, ±40 and ±80 s, for 1,320 inputs. Second-half coordinates are flipped and training data is augmented by swapping home and away. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- Models (scikit-learn 0.20.3), frame-wise test results: LR 86.7% acc / 88.7 F1; DT 84.6 / 87.2; RF 89.3 / 91.2; **AdaBoost 92.0% acc (±2.1) / 93.1 F1**, precision 93.4, recall 93.0. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- Stoppage-level results (stoppages ≥2 s, matched at IoU ≥50%), AdaBoost: precision 81.1, recall 78.9, **F1 79.8**. RF stoppage F1 is only 67.9. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- Boundary timing: 77% of starts and 81% of ends fall within ±2 s, and more than 51% of starts and 58% of ends within ≤1 s. Mean absolute shift is 1.79 ± 3.44 s (start) and 1.68 ± 3.47 s (end). Median shift is 1.00 s (start) and 0.84 s (end). The paper's own numbers are slightly inconsistent (77% vs "more than 78%"). — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- Post-processing: median filter of 91 frames (3.64 s), tuned on validation F1. This step matters a lot for segment quality. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- The paper cites earlier work by Gudmundsson and Wolle that extracted "ball-in"/"ball-out" events from player and ball data. That method is not verified here because it was found only second-hand. — [Search summary of Lang et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9522646/)

**Vidal-Codina, Evans, El Fakir, Billingham (2022), "Automatic event detection in football using tracking data", Sports Engineering / arXiv 2202.00804. No code link in the paper (Python 3, about 3 min per match).** [arXiv](https://arxiv.org/abs/2202.00804)
- Data: Tracab (FCWC19-20 via FIFA; 12 Bundesliga 2019-20 games via DFL), Track160 and Hawk-Eye, all at 25 Hz in 2D. The game status comes "either directly with a boolean that switches between in-play or dead ball (Tracab) or indirectly with missing ball data when the game is dead (Track160 and Hawk-Eye)". — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- Set-piece triggers are evaluated over each dead-ball interval [d0, dc]. A trigger must hold from some frame d1 until the last dead frame dc. — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
  - Kick-off: all players in their own half (tolerance εk1) and at least one player within εk2 of the centre mark (IFAB Law 8).
  - Penalty: one player on the goal line between the posts, one opponent at the penalty mark, everyone else outside the box and 9.15 m from the mark (Law 14).
  - Goal kick: at least one player inside their own goal area (Law 16).
  - Corner: at least one player within εc of an active corner mark (Law 17).
  - Throw-in: at least one player beyond the auxiliary sideline, i.e. sideline minus εt (Law 15).
  - Free kick: the fallback when no other pattern matches.
- Results: kick-offs and penalties are captured almost perfectly. Corner, throw-in and goal kick reach more than 90% precision and recall. Free kicks are worst because they have no pattern. Typical errors: throw-in or free kick near a corner classified as a corner; free kick near the sideline classified as a throw-in; free kick in the goal area classified as a goal kick. — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- The authors list as limitations that the method needs ball tracking and an in-play/dead flag. They suggest referee-whistle audio to set in-play/dead intervals more accurately, and a video classifier for edge cases such as drop ball vs free kick or corner vs throw-in. — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)
- Possession zone radius was 0.5 m for one provider and 1 m for the other two. The authors say a larger radius suits less accurate tracking (single tactical camera or broadcast). — [arXiv PDF](https://arxiv.org/pdf/2202.00804v3)

**Other tracking-based context**
- A 2023 European Journal of Sport Science study of 1,083 matches split tracking into ball-in-play and ball-out-of-play using event data. Distance-covered rate was more than 2x higher during BIP than BOP, which supports collective speed as a strong feature. — [search result summary; original not fetched](https://repository.mdx.ac.uk/item/8888v)
- A Springer chapter detects kicks from local maxima of ball acceleration and reports net-play-time detection precision and recall. Numbers not retrieved. — [Springer chapter PDF](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/978-3-319-24560-7_13.pdf)

### Inferences
- For our data, the Lang et al. recipe fits well, because it does not need the ball, which is missing in 41% of our frames. It needs 22 player positions with team (we have these) and labelled training matches. We have no labelled matches for LaLiga, but DFL open data (Section 4) gives 7 matches with a provider alive/dead flag. Domain shift (TRACAB vs Metrica automatic tracking, different noise) is a risk. Training on DFL plus a self-training or threshold-tuning pass on our match is plausible.
- Practical feature set for a GBM (LightGBM/XGBoost, faster than AdaBoost):
  - mean and median player speed and their rolling windows (stoppages show a sharp collective deceleration);
  - spread and compactness of each team;
  - number of players in own half;
  - min distance of any player to the corner flags, sidelines, goal area and centre spot (the Vidal-Codina triggers become features);
  - ball features when present: outside pitch bounds by a margin, speed about 0, stationary at a restart spot;
  - "ball missing" duration, used as a feature, not as the label;
  - Smart Tagging set-piece interval flags.
  Then median or HMM smoothing, with the minimum segment duration tuned for temporal IoU.
- Expected boundary error is about 1-2 s. At 25 fps that is 25-50 frames, which matters for temporal IoU. A boundary-refinement stage helps: snap the end of a stoppage to the first ball-kick frame (ball acceleration peak near the restarting player), and its start to the last frame the ball crosses the pitch line or a whistle/deceleration onset.

### Gaps
- I found no open-source code for Lang et al. or Vidal-Codina et al.
- I found no paper that reports alive/dead accuracy from automatic, uncorrected single-camera tracking (our setting). All reported numbers use TRACAB-grade multi-camera data.
- Gudmundsson & Wolle's ball-in/out method was not verified.

## 2. Video-based: SoccerNet action spotting and ball action spotting (classes, "ball out of play" results, licences, running on own video)

### Takeaway
SoccerNet-v2 action spotting has 17 classes, including "Ball out of play", "Throw-in", "Corner", "Kick-off", "Direct/Indirect free-kick" and "Clearance" (the goal kick). Its original baselines spot "Ball out of play" at 38.7-63.9 Average-mAP (loose tolerance, broadcast video). Newer models (NetVLAD++ on Baidu features at about 74 avg-mAP overall, E2E-Spot, T-DEED) are much stronger, but I found no per-class out-of-play numbers for them. These models are trained on broadcast footage with 1-60 s tolerances, so they spot a timestamp rather than precise dead-ball boundaries. On a tactical wide-angle camera they would face domain shift.

### Cited Findings
- SoccerNet-v2 has 110,458 annotated actions in 17 classes; it added classes such as ball out of play, clearances and penalties. — [SoccerNet 2024 results / search summary](https://arxiv.org/pdf/2409.10587), [SoccerNet-v2 paper](https://orbi.uliege.be/bitstream/2268/253781/1/SoccerNet-v2.pdf)
- **SoccerNet-v2 per-class test Average-mAP (%), Table 2:** — [ar5iv SoccerNet-v2](https://ar5iv.labs.arxiv.org/html/2011.13367)

  | Class | MaxPool | NetVLAD | AudioVid | CALF |
  |---|---|---|---|---|
  | Ball out of play (6,460 test instances) | 38.7 | 47.4 | 54.3 | **63.9** |
  | Throw-in | 34.7 | 42.4 | 50.0 | **56.4** |
  | Corner | 26.5 | 55.1 | 66.0 | **71.8** |
  | Kick-off | **30.3** | 45.7 | 52.9 | 37.2 |
  | Direct free-kick | 13.5 | 33.6 | **46.7** | 43.5 |
  | Indirect free-kick | 17.9 | 16.7 | 22.7 | **41.5** |
  | Clearance | 14.9 | 32.7 | 46.7 | **51.6** |

  The fetched summary bolded CALF for Direct free-kick, but the numbers show AudioVid higher (46.7 vs 43.5); the bold here follows the numbers.
- The same SoccerNet-v2 paper reports camera shot segmentation at 47.3% mIoU (CALF-seg) and replay grounding at 41.8% Average-AP (CALF). Both are relevant for spotting replays and close-ups, which correlate with stoppages in broadcast video. — [ar5iv SoccerNet-v2](https://ar5iv.labs.arxiv.org/html/2011.13367)
- NetVLAD++ (Giancola & Ghanem, CVPRW 2021) reaches 53.4% Average-mAP, +12.7 over CALF, and improves 16 of 17 classes, all except Goal. — [arXiv 2104.06779](https://arxiv.org/pdf/2104.06779)
- NetVLAD++ on Baidu soccer features reaches 74.1% avg-mAP on the SoccerNet-v2 test set (survey). — [Survey arXiv 2309.12067](https://arxiv.org/pdf/2309.12067)
- The SoccerNet pip package evaluation reports per-class mAP and introduced "tight" average-mAP in 2022. Per-class out-of-play numbers for newer models could be computed from released predictions. — [sn-spotting Evaluation](https://github.com/SoccerNet/sn-spotting/blob/main/Evaluation)
- **Ball Action Spotting 2024** (12 classes: Pass, Drive, Header, High Pass, **Out**, Cross, **Throw In**, Shot, Ball Player Block, Player Successful Tackle, **Free Kick**, Goal). The winner was **T-DEED** (Xarles, Escalera, Moeslund, Clapés), an ensemble of two models trained jointly on SoccerNet Ball and the original SoccerNet: **mAP@1 = 73.39** vs 71.35 runner-up and 56.15 baseline. Per-class "Out" was not reported on the page. — [SoccerNet 2024 Challenges Results](https://arxiv.org/html/2409.10587v1)
- **Team Ball Action Spotting 2025**: best Team-mAP@1 = 60.03 (team "dudek"), vs 51.72 for the T-DEED-based baseline. One review attributes the winner to int8.io re-implementing T-DEED with a joint team-action head; the attribution conflicts between sources. — [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182), [review](https://www.themoonlight.io/review/soccernet-2025-challenges-results)
- SoccerNet 2026 has no standalone action spotting task. Its five tasks are ball action anticipation, player-centric ball action spotting (FOOTPASS), novel view synthesis, Synloc and VQA. No "ball out of play" class was found. — [SoccerNet 2026 Challenges Results](https://arxiv.org/pdf/2607.07320), [FOOTPASS paper](https://arxiv.org/pdf/2511.16183)
- **Repos and licences** (GitHub API, checked 2026-10-28):
  - [SoccerNet/sn-spotting](https://github.com/SoccerNet/sn-spotting): **no licence file**, 105 stars, last push 2024-02. Contains CALF, NetVLAD++ and other baselines plus the devkit.
  - [SoccerNet/sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting): GPL-3.0, last push 2025-08 (2025 team BAS devkit, T-DEED-based baseline).
  - [arturxe2/T-DEED](https://github.com/arturxe2/T-DEED): GPL-3.0, 40 stars, last push 2026-01.
  - [jhong93/spot](https://github.com/jhong93/spot) (E2E-Spot): BSD-3-Clause, 88 stars, last push 2023-02.
  - [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate): GPL-3.0, 468 stars, active 2026-05.
- SoccerNet video data requires signing an NDA; the code repos are separate. Our own video can be run through E2E-Spot/T-DEED-style models: they take raw frames, typically resized to 224-398 px at 25 fps, with no precomputed features. — [general knowledge; NDA requirement per soccer-net.org, not re-fetched]

### Inferences
- For our task, video is most useful as a complement, for example in a per-frame classifier over tactical-camera frames. Ready-made SoccerNet spotters give events, not intervals, and were trained on broadcast footage. "Ball out of play" marks the moment the ball leaves; we would still need the restart moment, which T-DEED's "Throw In"/"Free Kick" classes or tracking-based kick detection can supply.
- A cheaper video route for a 3-week hackathon: train a small per-frame or clip classifier (e.g. a frozen image backbone plus a temporal 1D-conv/GRU) using our own tracking-derived pseudo-labels or labels from other matches. That avoids the broadcast-to-tactical domain gap. Feasibility depends on whether other labelled matches with video exist (the rules allow labelled data from other games).
- T-DEED (GPL-3.0) is the strongest open precise spotter for ball events at a 1 s tolerance. Fine-tuning it requires labelled clips. The GPL licence is fine for a hackathon but is copyleft if the code is redistributed.

### Gaps
- I found no per-class "Ball out of play" results for NetVLAD++, Baidu, E2E-Spot, T-DEED or COMEDIAN.
- I found no BAS 2024 per-class "Out" number.
- COMEDIAN's repo and licence were not verified.
- I found no reports of SoccerNet-trained spotters applied to tactical, non-broadcast cameras.

## 3. Game-state / replay / camera-shot classification and other cues for stoppages

### Takeaway
SoccerNet-v2 camera-shot segmentation and replay grounding exist (47.3% mIoU; 41.8% avg-AP), but they target broadcast video. A tactical wide-angle feed has no replays or camera cuts, so those cues do not transfer. The only other cue found is referee-whistle audio, which the Vidal-Codina authors recommend. I found no open soccer whistle dataset.

### Cited Findings
- Camera shot segmentation is 47.3% mIoU (CALF-seg) and replay grounding is 41.8% Average-AP. — [ar5iv SoccerNet-v2](https://ar5iv.labs.arxiv.org/html/2011.13367)
- SoccerNet Game State Reconstruction (tracking plus identification on a minimap): the 2024 winner (Constructor.tech) reached GS-HOTA 63.81 vs a baseline of 23.36. The code is [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) (GPL-3.0). — [SoccerNet 2024 Results](https://arxiv.org/html/2409.10587v1)
- The authors suggest whistle audio to establish in-play/dead intervals more accurately. — [Vidal-Codina et al.](https://arxiv.org/pdf/2202.00804v3)
- Whistle detection exists mainly in RoboCup work: a dataset recorded from robots, imbalanced about 10:1, with public availability unclear. There are older audio-visual highlight papers. I found no clearly open soccer whistle dataset. — [RoboCup referee paper arXiv 2411.17347](https://arxiv.org/pdf/2411.17347), [HULKs audio](https://hulks.de/hulk/robotics/perception/audio/), [Soccer highlights audio-visual arXiv 1411.6496](https://arxiv.org/pdf/1411.6496)

### Inferences
- If our tactical video has an audio track, a simple band-pass energy detector around the whistle fundamental (about 2-4 kHz) is a cheap feature for stoppage start and end. This needs checking on our file, and crowd noise may interfere.
- Our wide-angle tactical video makes the ball-out-of-bounds geometry (via homography) and the player configuration more observable than in broadcast. That favours tracking or geometric methods over SoccerNet broadcast models.

### Gaps
- I did not verify whether the hackathon video includes audio.
- I found no paper on stoppage detection from tactical (non-broadcast) video.

## 4. Effective playing time / ball-in-play estimation literature and PySport tooling with ball status

### Takeaway
DFL/Sportec open tracking (7 Bundesliga matches, 25 fps, CC-BY 4.0) carries a per-frame alive/dead ball status that floodlight, kloppy and databallpy expose. It is the best public labelled data for training or validating an alive/dead classifier on matches other than ours. The Metrica open sample data also uses dead/alive in databallpy's examples.

### Cited Findings
- **IDSSE / DFL open data** (Bassek, Rein, Weber, Memmert, Scientific Data, Feb 2025): 7 complete Bundesliga 1 and 2 matches from 2022/23, with TRACAB/Sportec tracking at 25 fps, events and metadata. Licence CC-BY 4.0, owner DFL. Match IDs: J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WPF, J03WQF. Load with `kloppy.sportec.load_open_tracking_data(match_id=...)`. — [PMC paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/), [HF dataset card](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- kloppy's Sportec loader has `only_alive=True` by default, so it **drops dead-ball frames**. Set `only_alive=False` to keep them, which confirms that a per-frame alive/dead status exists in the DFL data. — [kloppy Sportec reference (deploy preview)](https://deploy-preview-546--unruffled-nobel-ed43d4.netlify.app/reference/providers/sportec)
- kloppy tracking `Frame` has `ball_state` (BallState: alive/dead) and `ball_owning_team`. kloppy is BSD-3-Clause, actively maintained (push 2026-10). — [kloppy Frame (deploy preview)](https://deploy-preview-546--unruffled-nobel-ed43d4.netlify.app/reference/domain/models/tracking/frame), [GitHub](https://github.com/PySport/kloppy)
- floodlight `io.dfl.read_position_data_xml` returns `(xy_objects, possession_objects, ballstatus_objects, teamsheets, pitch)`. `ballstatus_objects[segment]` is a Code holding dead/alive. floodlight is MIT. — [floodlight DFL docs](https://floodlight.readthedocs.io/en/latest/modules/io/dfl.html), [GitHub](https://github.com/floodlight-sports/floodlight)
- databallpy's tracking schema has a `ball_status` column restricted to "alive"/"dead" (nullable). It warns if the ball is alive for less than half of the non-break frames, and uses ball status for event-tracking synchronisation. In the Metrica example data, the first and last frames have NaN ball_x and status "dead". databallpy is MIT, active (push 2026-10). — [databallpy schema source](https://databallpy.readthedocs.io/en/latest/_modules/databallpy/schemas/tracking_data.html), [databallpy example](https://databallpy.readthedocs.io/en/main/example.html), [GitHub](https://github.com/alek050/databallpy)
- Tracab-style providers give an explicit in-play boolean, while Track160 and Hawk-Eye signal dead ball by missing ball data. — [Vidal-Codina et al.](https://arxiv.org/pdf/2202.00804v3)
- Effective playing time per match is typically under 60 min (Wass et al. 2020; Mernagh et al. 2021, as summarised in search). These studies use ball-in-play splits but do not detect them. — [MDX repository](https://repository.mdx.ac.uk/item/8888v)
- ELASTIC (KAIST; CIKM 2026) synchronises events to tracking using tracking-only features. It provides a frame-by-frame benchmark on 3 Sportec open matches, and code is at github.com/hyunsungkim-ds/elastic (workshop version). It is useful for snapping restart events (kick frames) to tracking frames. — [arXiv 2608.30227](https://arxiv.org/pdf/2608.30227), [arXiv 2508.09238](https://arxiv.org/html/2508.09238v1)

### Inferences
- Concrete training plan:
  1. Load the 7 DFL matches with kloppy (`only_alive=False`) to get per-frame ball_state labels.
  2. Degrade the DFL data to mimic our input: randomly drop ball positions in about 41% of frames, concentrated in crowded and dead phases, and add position noise.
  3. Train a Lang-style player-centric GBM plus ball features.
  4. Apply it to the LaLiga match and tune smoothing and thresholds without hand labels, for example by checking that predicted alive time is about 55-60 min and that Smart Tagging set-piece intervals fall inside dead segments.
- Caveat: the DFL ball-status flag reflects the provider's definition. Our ground truth comes from "corrected tracking" by Metrica and may differ at boundaries, e.g. whether the ball is alive the instant a throw-in leaves the hands. The match-wide alive share in our ground truth should be checked against DFL's before relying on it.

### Gaps
- I did not verify the exact encoding of the DFL BallStatus values in the raw XML (likely 1 = alive, 0 = dead).
- I found no Opta or StatsPerform public method for automatic dead-ball detection.
- The Metrica open sample data (github.com/metrica-sports/sample-data) was not checked for a per-frame ball-status field. It has events with set-piece types that could act as weak labels.

## 5. What works best and is practical in a ~3-week hackathon

### Takeaway
The priority order is: (1) rules and features from tracking geometry (Vidal-Codina triggers, ball out of bounds via homography, collective speed drop, ball stationary at restart spots) combined with Smart Tagging intervals; (2) a supervised GBM trained on DFL open data with an alive/dead flag, using Lang-style multi-scale temporal features; (3) boundary refinement and temporal smoothing tuned to temporal IoU; (4) optional video/audio cues. Pretrained SoccerNet spotters are lower priority because of the broadcast domain gap and the timestamp-only outputs.

### Cited Findings
- Player-only ML gives about 92% frame accuracy with about 1-2 s boundary error on TRACAB data. — [Lang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- Rule-based restart triggers reach more than 90% precision and recall for corner, throw-in and goal kick, and near-perfect results for kick-off and penalty, given the dead interval. — [Vidal-Codina et al.](https://arxiv.org/pdf/2202.00804v3)
- Median filtering of about 3.6 s improved F1 in Lang et al. — [Lang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)
- The best open precise video spotter is T-DEED: BAS 2024 mAP@1 of 73.39, GPL-3.0. — [SoccerNet 2024 Results](https://arxiv.org/html/2409.10587v1), [T-DEED repo](https://github.com/arturxe2/T-DEED)

### Inferences (practical recipe, prioritized)
1. **Day 1-3: fix the baseline.** Stop treating "ball missing" as dead. Interpolate or hold the ball through short gaps, and assign dead only when the evidence below agrees.
2. **Geometric dead-ball evidence:**
   - the last ball position before a gap or loss is outside the pitch lines (homography-normalised, margin about 0.5-1 m);
   - ball speed about 0 for more than 1-2 s at a restart location (corner arc, goal area, sideline, centre spot, or any static position with a player adjacent);
   - Vidal-Codina configuration triggers: a player beyond the sideline, at a corner, in the goal area, or in the kick-off formation.
3. **Collective dynamics:** the team-average speed drop and a ball-independent model on ±0.4-80 s windows (Lang et al.) cover the frames without a ball.
4. **Smart Tagging:** set-piece intervals are strong priors for dead segments. Use their end, i.e. the restart kick, as a boundary anchor and refine it to the ball-kick frame (ball acceleration peak near a player). Throw-in team can be inferred from which side's player stands beyond the sideline, or which team touches the ball first after it.
5. **Model:** a LightGBM per-frame classifier on (2)+(3)+(4) features, trained on degraded DFL open data (7 matches, CC-BY), followed by an HMM or median smoothing with a minimum duration. Tune on DFL hold-out data for segment-IoU, not frame accuracy.
6. **Boundary precision:** dead start = the frame the ball crosses a line, or the foul and whistle onset (speed drop). Dead end = the restart kick frame. Every second of boundary error splits or merges possession segments, so this stage likely gives the biggest metric gain.
7. **Optional:** whistle-audio energy detector if audio exists, and a small frame classifier on tactical video for edge cases (injuries, VAR, substitutions with the ball stationary in the middle).

### Gaps
- No published benchmark matches our exact metric (possession-segment temporal IoU) or input quality. Gains need to be validated internally on DFL hold-out data or on other Metrica-labelled matches if provided.
