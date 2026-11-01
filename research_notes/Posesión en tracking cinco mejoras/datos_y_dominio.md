# Additional open data, label strategies and domain adaptation for football possession models

Research date: 2026-11-01. Scope: make a model trained on degraded IDSSE transfer better to the challenge's Metrica tracking. This extends `research_notes/Posesión de balón en tracking/datasets_abiertos.md`.

Items tagged **[measured]** were computed in this session from open data and the repo's local IDSSE caches, using simple scripts. They are direct measurements, but the rules are first-pass heuristics, not validated labels. Dead-ball and spell rules are described where used. The only public-excerpt facts used are the ones in the assignment and DATA.md (17 possessions, mean 14.2 s, class counts). Nothing was fitted to them. No file about the challenge match's labels was opened.

## Q1. Which open tracking datasets can supply possession labels, and are they usable?

### Takeaway
No new large dataset beats the existing plan. The most valuable addition is the small **Metrica sample set (3 matches, about 183 MB)**: same provider, and it publishes that provider's own possession definition. In Games 1 and 2, a missing ball in the corrected tracking is almost exactly dead ball. SkillCorner is the second addition. PFF has 64 matches, but its terms could not be read, so it is blocked until they are.

### Cited Findings

| Dataset | Size verified | Licence | Possession / dead labels | Quality | Verdict |
|---|---|---|---|---|---|
| IDSSE/DFL | 7 matches. Tracking XML 349–419 MB each (about 2.62 GB total), events 0.6–0.8 MB each, meta 12 KB each — [Figshare API](https://api.figshare.com/v2/articles/28196177). Mirror 2.63 GB, 23 files — [HF](https://huggingface.co/datasets/pysport/idsse-data) | CC BY 4.0 — [Figshare API](https://api.figshare.com/v2/articles/28196177) | Native per frame | TRACAB optical | Already in use |
| Metrica sample | Game 1: 65.6 MB tracking, 139 KB events. Game 2: 61.5 MB, 154 KB. Game 3: 53.4 MB, 1.7 MB events, 82 KB metadata — [GitHub tree](https://api.github.com/repos/metrica-sports/sample-data/git/trees/master?recursive=1) | None named. Asks for responsible use and source credit — [README](https://github.com/metrica-sports/sample-data) | Derive from events; **dead derivable from ball NaN** (see below) | Described as showing the Elite package, so treat as corrected, not automatic — [README](https://github.com/metrica-sports/sample-data) | **Add** |
| SkillCorner | 10 A-League 2024/25 matches at 10 fps — [README](https://github.com/SkillCorner/opendata). One tracking file is 89.5 MB [measured], so about 0.9 GB in total (extrapolated) | MIT at repo level — [GitHub API](https://api.github.com/repos/SkillCorner/opendata); README asks for credit | Per-frame `possession.group` — [README](https://github.com/SkillCorner/opendata) | Broadcast, automatic. `is_detected` flags distinguish detected positions from extrapolated ones — [kloppy #639](https://github.com/PySport/kloppy/issues/639) | Add as secondary |
| PFF / Gradient WC22 | 64 matches, `{game_id}.jsonl.bz2`; later restructured by game — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data); [X post](https://x.com/Gradient_Sports/status/1890045302989832275). **Size not found** | **Unverified.** The form reportedly implies accepting Terms of Use, but the text could not be retrieved. [fc.pff.com](https://fc.pff.com/) returned HTTP 530 and the [terms page](https://www.gradientsports.com/terms-conditions) came back without body text | kloppy reads `game_event.home_ball` and `game_event_type`, and sets dead only on BALL_OUT or PERIOD_END — [kloppy source](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py) | Broadcast; smoothed fields are what kloppy reads | **Blocked on terms** |
| SoccerTrack v2 | HF `usedStorage` 256.9 GB, auto-gated — [HF API](https://huggingface.co/api/datasets/atomscott/soccertrack-v2) | CC BY 4.0 | Would require video pipeline work | Amateur/university, video | Reject (86% of available disk) |
| TeamTrack | 2 soccer recordings of 30 min each — [project page](https://atomscott.github.io/TeamTrack/) | Code MIT — [GitHub API](https://api.github.com/repos/AtomScott/TeamTrack). Dataset licence not stated. The Kaggle link gave 404 to a scripted request | No ball annotation mentioned | Bounding boxes | Reject |
| SportsMOT | 240 videos, people only — [HF card](https://huggingface.co/datasets/MCG-NJU/SportsMOT) | CC BY-NC 4.0, redistribution needs permission — [GitHub](https://github.com/MCG-NJU/SportsMOT) | None | Boxes | Reject |
| Last Row | 19 Liverpool goals, no events — [related repos summary](https://relatedrepos.com/gh/Friends-of-Tracking-Data-FoTD/Last-Row); no licence reported by [GitHub API](https://api.github.com/repos/Friends-of-Tracking-Data-FoTD/Last-Row) | None reported | None | Goals only | Reject |
| Soccermatics | An R visualisation package, not a dataset — [GitHub](https://github.com/JoGall/soccermatics) | n/a | n/a | n/a | Not a data source |

**Metrica sample, provider-specific findings**
- Metrica publishes its definition: a player has possession "as soon as they touch the ball" and keeps it until another player takes over, a shot is deflected, the referee significantly deflects it, or it goes out. Possession "is not considered lost when an opposing player's interference does not have a significant impact" — [Metrica event definitions](https://github.com/metrica-sports/sample-data/blob/master/documentation/events-definitions.pdf)
- That wording closely resembles the challenge rule that touches and deflections are not changes — [CHALLENGE.md](../../CHALLENGE.md)
- The tracking has no owner or status column. kloppy hardcodes both to `None` for Metrica CSV — [kloppy source](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/metrica_csv.py). So the "owning" field used by the challenge cannot be reproduced from open data.
- **[measured] Ball NaN is dead ball.** The test used the home-team tracking files.

| Check | Game 1 | Game 2 |
|---|---|---|
| Frames with ball NaN | 39.1% of 145,006 | 41.0% of 141,156 |
| PASS inside a NaN gap | 0/799 | 0/964 |
| SHOT inside a gap | 0/24 | 0/24 |
| RECOVERY inside a gap | 0/278 | 0/248 |
| BALL LOST inside a gap | 0/257 | 0/233 |
| CARD inside a gap | 4/4 | 6/6 |

- Also in Game 1 [measured]: 51/51 BALL OUT events lie within 2 s of a gap start, and 75/77 SET PIECE events within 2 s of a gap end. Gap length is p10 0.04 s, median 7.8 s, p90 34.6 s. Ball-present runs average 22.8 s.
- Metrica sample files date from at least April 2020 — [commit history](https://api.github.com/repos/metrica-sports/sample-data/commits?path=data/Sample_Game_1/Sample_Game_1_RawEventsData.csv&per_page=1). Game 3 was added in 2021 per the prior notes. Neither can contain a 2024-25 match, even though the games are anonymised.

**SkillCorner, findings**
- kloppy sets ball state to ALIVE only when `possession.group` is a team, otherwise DEAD — [kloppy source](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/skillcorner.py). So kloppy's dead flag is inferred, not provider-stated.
- [measured, match 2017461 only] 97.4 min, null group on 50.6% of frames, ball coordinates on 69.1%. 326 spells, mean 8.9 s, median 6.1 s, 22% under 2 s. Null runs: 118, mean 25.1 s, median 18.7 s.
- One community report found 40.7% of player records and 17.7% of ball positions extrapolated in one match — [kloppy #639](https://github.com/PySport/kloppy/issues/639)

**Correction to earlier notes:** the seven IDSSE IDs are J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 — [Figshare API](https://api.figshare.com/v2/articles/28196177). `datasets_abiertos.md` lists J03WPF and J03WQF, which are wrong. [possession/idsse.py](../../possession/idsse.py) already has the right IDs.

No 2026 open tracking release with frame-level possession was found. Searches for one, including a Gradient 2026 release and Signality open data, returned nothing relevant — [open-football index](https://github.com/withqwerty/open-football).

### Inferences
- **Ball NaN as the corrected dead mark is a strong working hypothesis for the challenge.** The challenge says truth follows what the tracking marks as dead — [CHALLENGE.md](../../CHALLENGE.md). The sample shows how Metrica's corrected tracking appears to do this. That is an inference from sample files, not a statement about the challenge file.
- The 39–41% NaN rate in corrected samples resembles the 41% missing rate in the challenge's automatic tracking. This may be coincidence, since DATA.md says missing ball is concentrated on crowded and aerial frames, and D is only about 20% of the 5 public minutes. It should not be read as "missing means dead".
- It does fit the repo's `dead41` experiment, where placing gaps on dead ball scored +0.051 in 7 of 7 matches — [plan](../../docs/plan/continuar_m4_gpu.md).
- Metrica's value is less in the row count than as a **rehearsal target**: the only open data with the target's provider, coordinate format and possession language.
- Licence handling for Metrica: keep the data out of the repo, download it in a script, and credit it. The absence of a named licence is a real ambiguity. Ask the organisers if the final solution depends on it. The same no-redistribution approach suits SkillCorner and any PFF use.

### Gaps
- Metrica Game 3 was not tested for the NaN pattern (different file format).
- No open automatic Metrica tracking was found, so the corrected-to-automatic step cannot be calibrated on open data.
- PFF licence, size, exact fps and meaning of `home_ball` remain unverified. The dataset was not downloaded.
- SkillCorner's meaning of null `possession.group` (dead versus uncontested) is unconfirmed. The per-match result is from one match. `phases_of_play` is defined only in play — [README](https://github.com/SkillCorner/opendata) — and could cross-check it, but this was not tested.

## Q2. Which event-tracking synchronisation methods can derive labels?

### Takeaway
ELASTIC is clearly the best choice (MPL-2.0, 96.5% of event starts within 2 frames, about 38 s per match). It applies only to matches with events, so it serves IDSSE label cleaning. Metrica samples are already synchronised per their README.

### Cited Findings
| Tool | Licence | Event-start accuracy within 2 frames (W2) | Notes |
|---|---|---|---|
| ELASTIC-NW | MPL-2.0 — [GitHub API](https://api.github.com/repos/hyunsungkim-ds/elastic) | 96.5%; event ends 93.6% — [Kim et al., CIKM'26](https://arxiv.org/pdf/2608.30227) | Mean difference 1.697 frames overall; 37.6 s per match |
| DataBallPy | MIT — [GitHub API](https://api.github.com/repos/Alek050/databallpy) | 74.8%, on a subset of event types — [Kim et al.](https://arxiv.org/pdf/2608.30227) | 4.0 s per match. [Docs](https://databallpy.readthedocs.io/en/latest/getting_started/synchronisation_page.html) give no validation figure |
| Biermann et al. | not checked | 67.1%, subset of event types — [Kim et al.](https://arxiv.org/pdf/2608.30227) | Passes-focused — [Sci Reports](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10518005/) |
| ETSY | Apache-2.0 header — [LICENSE](https://raw.githubusercontent.com/ML-KULeuven/ETSY/main/LICENSE) | 54.4% — [Kim et al.](https://arxiv.org/pdf/2608.30227) | Rule-based — [KU Leuven](https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization) |
| sync.soccer | AGPL — [GitHub](https://github.com/huffyhenry/sync.soccer) | No figure published | Opta F24 + Tracab only; 5–10 min and about 10 GB RAM per game; proof of concept |

- The ELASTIC benchmark was built by re-annotating timestamps on the Sportec Open DFL Dataset, the same matches as IDSSE — [Kim et al.](https://arxiv.org/pdf/2608.30227). All accuracy figures come from the ELASTIC authors, who built the benchmark.
- ELASTIC detects event ends, meaning ball receptions that the raw data do not record — [Kim et al.](https://arxiv.org/pdf/2608.30227). Those are direct evidence of controlled reception.
- PathCRF's label pipeline on the same data: ELASTIC sync, then RDP ball-direction changes as touch candidates, aligned by Needleman-Wunsch — [PathCRF](https://arxiv.org/pdf/2602.12080).
- Metrica's tracking and events are stated to be synchronised — [README](https://github.com/metrica-sports/sample-data).

### Inferences
- ELASTIC should replace the crude kick-off alignment used in Q3 and allow a window far narrower than ±1.5 s.
- Do not run any of this on the challenge match. It needs event data, and any such event data would be human-made.

### Gaps
- ELASTIC's published accuracy is on TRACAB-quality data. Nothing was found on degraded or automatic tracking.
- Whether the re-annotated timestamps are downloadable was not confirmed.

## Q3. How can IDSSE's flicker-prone possession be converted to a control-based label?

### Takeaway
Require evidence of control for each team change. **[measured]** Keeping a `BallPossession` spell only if its team has a Play or ShotAtGoal event nearby reduces spells from 437 to 227 per match and raises the mean from 7.2 s to 13.9 s, relabelling only 4.9% of frames. Most of the 7.2 s vs 14.2 s gap looks like label definition and small-sample noise rather than different football.

### Cited Findings
- Link & Hoernig separate Team Ball Possession (56:04 ± 5:12 min per match) from Team Playmaking (50:01 ± 7:05) and Team Ball Control (17:49 ± 8:13). Control needs a choice of play options. Headers after crosses and midfield ping-pong usually count as possession but not control — [PLOS ONE 2017](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953)
- In that study, interval classification reached κ = .67 against human inter-rater κ = .72, so control is only partly objective — [PLOS ONE 2017](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953)
- IDSSE's `BallPossession` is based on the player in control per the dataset paper — [Bassek et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/). The measured flicker below suggests that in practice it reacts to brief contacts.
- Zone-based definitions, such as a circle of radius R around each player with duel rules, are common alternatives — [Vidal-Codina et al.](https://arxiv.org/pdf/2202.00804)
- Published providers differ in possession figures for the same match, per a low-authority overview — [Medium](https://medium.com/@davidleefindlay/how-is-possession-calculated-in-football-4ed063fac418)

**[measured] IDSSE, 7 matches, spells as maximal H or A runs.** Windows are 5 minutes, stepped every 60 s. Their p5–p95 column shows the spread of mean duration, and the last column the share of windows with mean ≥ 14.2 s.

| Label | Spells/match | Mean s | Median s | p90 s | Under 2 s | Window p5–p95 | P(≥14.2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw `BallPossession` | 437 | 7.2 | 3.7 | 18.6 | 30% | 4.1–13.9 | 4.7% |
| Merge A-B-A under 1 s | 386 | 8.2 | 4.4 | 20.5 | 23% | 4.6–15.0 | 6.6% |
| Merge under 2 s | 323 | 9.8 | 5.7 | 24.0 | 15% | 5.3–17.8 | 15.4% |
| Merge under 3 s | 278 | 11.4 | 6.8 | 26.7 | 15% | 6.1–20.5 | 25.3% |
| Merge under 5 s | 228 | 13.8 | 8.4 | 32.0 | 14% | 7.2–25.4 | 42.1% |
| Events only (last Play/Shot team) | 247 | 12.8 | 8.2 | 29.8 | 11% | 7.2–22.0 | 34.7% |
| **Hybrid: raw spell kept only if its team has Play/Shot within ±1.5 s** | **227** | **13.9** | **9.2** | **30.6** | **6%** | 7.4–23.4 | 41.0% |

- Method details: dead frames come from `BallStatus` and are untouched. Events were aligned only by matching kick-off times per half. The event team equals `BallPossession` on 89.7–94.0% of alive event frames, by match. The events-only variant agrees with raw on about 80% of alive frames.
- IDSSE alive share is 55.1% (48–59% by match), and alive intervals average 31.4 s.

**[measured] Metrica samples, same family of rule (event-only).**

| Game | Dead from | Team flips on | Spells | Mean s | Median s | Under 2 s |
|---|---|---|---:|---:|---:|---:|
| 1 | Events | PASS/SHOT/BALL LOST | 326 | 11.4 | 6.5 | 13% |
| 2 | Events | same | 311 | 12.6 | 8.0 | 14% |
| 3 | Events | same | 365 | 11.5 | 6.6 | 15% |
| 1 | Ball NaN | same | 345 | 10.2 | 5.3 | 19% |
| 1 | Ball NaN | same + RECOVERY | 402 | 8.8 | 3.9 | 28% |

With events-based dead and RECOVERY included, means are 9.8, 10.8 and 10.3 s.

### Inferences
- **Under event-only rules, IDSSE (12.8 s) and Metrica (11.4–12.6 s) nearly agree.** That suggests provider labelling differences account for much of the gap to raw IDSSE. The event taxonomies differ, so this is suggestive only.
- **14.2 s comes from only 17 possessions.** It would be rare but not impossible even under raw labels (4.7% of windows), and ordinary under control labels (35–41%). Barcelona's style may also contribute. Windows overlap and truncated spells pull means down, so these shares are approximate.
- **Duration alone cannot choose the label.** Merge-under-5 s and the hybrid give about the same mean, but very different short-spell shares. Merging by duration is blunt: it erases real short possessions, and the spell-matching metric will punish that. Evidence-based rules are defensible because they follow the challenge's definition.
- **Do not pick the rule to hit 14.2 s.** That would be tuning on the public excerpt. The ±1.5 s window here was chosen before seeing results and was not swept. The right selection test is agreement with Metrica sample labels or other non-challenge data.
- Recommended label pipeline: ELASTIC sync, then hybrid filter, then optionally RDP/ELASTIC reception evidence. Keep raw labels as an auxiliary target, because a two-head model may learn more from both than from relabelling alone. Handle aerial duels explicitly: TacklingGame and OtherBallAction events were excluded from the evidence set here because they do not prove control.
- Change-of-possession training weight should fall on hybrid-confirmed changes only, since raw flicker teaches the model the wrong kind of transition.

### Gaps
- No published study comparing possession definitions by their effect on duration distributions was found; the table above is the closest evidence and is unreviewed.
- Hybrid labels have not been tested for their effect on spellF1. The relabelled 4.9% was not inspected by eye.
- The kloppy Sportec loader maps any `BallPossession` other than 1 to away — [kloppy source](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/sportec/deserializer.py). Alive frames with a missing value would go to away. How often this occurs was not checked.
- Metrica's "owning" field in the challenge may not follow the Metrica event definition. These notes have no way of confirming that.

## Q4. What evidence supports domain adaptation, importance weighting, test-time adaptation and pseudo-labelling here?

### Takeaway
Cross-provider loss is real and large, but no sports trajectory study of unsupervised adaptation was found. Evidence comes from vision and general time series, and warns mainly about pseudo-label errors and model selection. Adaptation should be adopted only if it wins a rehearsal on Metrica samples with fixed settings.

### Cited Findings
- **Shift size in football:** PathCRF event F1 was 73.88% within Sportec and 79.83% within K League, but 49.16% for Sportec→K League and 66.25% for K League→Sportec. K League is 10 Hz GPS and 15 matches — [PathCRF, Appendix B](https://arxiv.org/pdf/2602.12080).
- **More training data matters:** the authors attribute K League's higher within-dataset score to its larger training set — [PathCRF](https://arxiv.org/pdf/2602.12080).
- **Vendor differences:** varying event definitions across vendors create inconsistencies — [Springer 2025](https://link.springer.com/article/10.1007/s12283-025-00533-4).
- **AdaTime** (11 methods, 5 datasets, 50 scenarios): carefully tuned visual adaptation methods are competitive with time-series-specific ones. Model selection strongly affects results, and many earlier methods used target labels to select, violating the premise — [Ragab et al.](https://arxiv.org/abs/2203.08321)
- **Label shift:** BBSE estimates target class priors from a black-box model's confusion matrix — [Lipton et al.](https://arxiv.org/pdf/1802.03916.pdf). Saerens EM needs no retraining but assumes calibrated probabilities. Calibrating on held-out source data first makes EM a strong baseline — [Alexandari et al.](https://arxiv.org/abs/1901.06852v4)
- **Test-time adaptation:** error accumulates on long sequences — [Marsden et al.](https://arxiv.org/pdf/2208.07736). Unreliable pseudo-labels and unstable optimisation are the main failure modes, addressed by graph-based correction and parameter averaging — [Ma, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Improved_Self-Training_for_Test-Time_Adaptation_CVPR_2024_paper.pdf). Results under one corruption protocol may not generalise — [Continual TTA survey](https://arxiv.org/pdf/2607.08164).
- **Pseudo-labelling** is established but sensitive to noise — [UDA review](https://www.researchgate.net/publication/372896834_Pseudo_Labels_for_Unsupervised_Domain_Adaptation_A_Review); [person re-ID paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12258581/).
- **Fine-tuning on target with labels** raised SoccerNet tracking HOTA from 43.9% to 57.9%, a vision result that shows the upper bound of adaptation rather than an unsupervised one — [SoccerNet-Tracking](https://www.alphaxiv.org/abs/2204.06918).

### Inferences
Ordered by expected value against risk:

1. **Rehearsal protocol first.** Train on degraded IDSSE, apply to degraded Metrica samples, and score against event-derived labels (dead from ball NaN). Compare no adaptation against each method with fixed settings. This addresses AdaTime's selection problem, with no challenge labels involved.
2. **Add Metrica samples to training** with ball-NaN dead labels. This is plain supervised learning, so it has no adaptation risk. The dead-ball boundary is probably the most transferable gain.
3. **Prior adjustment by calibrated EM.** Cheap and uses no retraining. But label shift assumes p(x|y) is stable, which this domain breaks, and alive share varies (55% IDSSE, about 60% Metrica samples, 80% in the 5 public minutes). Bound the priors, and adopt only if rehearsal agrees.
4. **Covariate weighting.** Train a discriminator between degraded-source and target feature windows, and use clipped density-ratio weights. If the discriminator separates them almost perfectly, the degradation is unrealistic. That diagnostic may be worth more than the weights. Track effective sample size.
5. **One or two rounds of conservative self-training.** Use high-confidence frames in the middle of long decoded spells only, and exclude transitions, which are exactly where errors concentrate and what spellF1 rewards. Keep source labels in the training set, and fix thresholds in advance. Decoder-level pseudo-labels are better than frame-level ones because of the temporal structure.
6. **Unsupervised decoder fit** (HMM/CRF transition rates via Baum-Welch on the target). Low parameter count makes this attractive, but it can drift towards flicker. Bound the rates using Metrica-sample spell statistics.

Avoid gradient-reversal and other adversarial approaches for now. The evidence is weak, it needs neural features, and one target match with no labels provides no way to select a stable model.

The rule check also matters. These steps are only allowed if no human decides labels, so the pipeline must be fully automatic with thresholds fixed from other matches. The public excerpt must not choose between variants.

### Gaps
- No study of unsupervised domain adaptation for football or sports trajectories was found.
- No time-series-specific test-time adaptation benchmark or survey was found, and no time-series evidence on confirmation bias.
- No evidence was found for any of these methods with gradient-boosted models on sequence features. The ranking above is judgement, not a reported result.
- One target match gives a small effective sample. It is unclear whether the adaptation gain would be measurable.

## Q5. How should clean tracking be degraded to simulate automatic or broadcast errors?

### Takeaway
No validated published noise model for automatic football tracking was found. Current `degrade()` places ball gaps uniformly at random, which contradicts the data. Replace it with a state-conditional gap model, plus tracklet fragmentation and a range of noise levels, calibrated on the unlabelled automatic input.

### Cited Findings
- TRACAB Gen5 position RMSE is 0.08 m, rising with object speed. Validation does not transfer across systems — [Linke et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064167/)
- The same paper cites 56 cm for previously tested video tracking systems — [Linke et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064167/)
- `degrade()` hides random stretches of the ball, drops random player slots, and adds smoothed noise — [train.py](../../possession/train.py). The plan already flags that gaps should sit on specific situations — [plan](../../docs/plan/continuar_m4_gpu.md)
- **[measured]** Corrected Metrica gaps on Game 1 have median 7.8 s and p90 34.6 s, and sit on dead ball (Q1). This is a corrected-data pattern, not automatic tracking.
- Ball Radar reaches 1.31 m and 93% team possession accuracy with 20% of the ball visible — per [prior notes](../Posesión%20de%20balón%20en%20tracking/eventos_y_balon_ausente.md)
- TranSPORTmer treats missing data by an input mask, unifying imputation and ball inference — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- A related football imputation study uses a camera module to simulate off-screen players on ground-truth matches — [TranSPORTmer search context; likely Sci Reports 2022](https://www.nature.com/articles/s41598-022-12547-0). The paper's identity is inferred from a search summary, so check it before citing.
- SkillCorner's `is_detected` flags offer a real broadcast pattern of which players and ball are missing — [README](https://github.com/SkillCorner/opendata)
- A 2026 paper compares broadcast-derived player and ball tracking against TRACAB Gen5 — [Sports Engineering 2026](https://link.springer.com/article/10.1007/s12283-026-00549-4). Its numbers could not be accessed (paywall redirect), and it should be read.

### Inferences
Proposed degradation design:
1. **Ball gaps by state.** Sample gap probability from a mixture over situation: dead ball, crowded area (player count near ball), aerial phase (ball height, available in IDSSE), high ball speed, boundary area. Fit the mixture so the marginal is 41% and gap-length, and gap-given-context, histograms match the automatic input. These are statistics of machine-made input, not labels. Confirm this reading against RULES.md.
2. **Tracklets, not slots.** Break tracks at occlusions, reassign IDs inside close groups, and include wrong-team assignment. The target has no player identities, so the model should never see stable slots.
3. **Position noise.** Scale with speed and with crowding rather than constant. Time-correlated error is right (the repo's rationale on phantom speed holds). The size should come from a target-side measure, such as jitter in stationary players at dead ball.
4. **Domain randomisation.** Train across a range of severities, not one point estimate, since the true level is uncertain.
5. **Discriminator check.** Use Q4's source-versus-target discriminator as the realism metric, feature by feature.
6. **Rehearse on Metrica.** Apply the same degradation to samples before scoring, so that the corrected-to-automatic step exists in rehearsal.

Note the risk in point 1: if the model learns "ball missing means dead" from degradation that over-concentrates gaps on dead ball, it will copy the failure the presentation already warns of. The target's gaps fall on interesting live frames too.

### Gaps
- No published broadcast noise model validated in metres against optical data was found.
- The degree to which automatic Metrica ball loss depends on state was not measured here. It needs the challenge input and is left to the team.
- The Sports Engineering 2026 numbers are unread.

## Q6. Which actions look most likely to improve transfer?

### Takeaway
Fix the labels and rehearsal before adding more data. Deadline is 17 Nov 2026, so PFF and neural adaptation should wait.

### Cited Findings
- spellF1 matches possessions one-to-one by temporal IoU, and possessions are maximal H or A runs — [EVALUATION.md](../../EVALUATION.md). So spell count and boundaries matter more than frame accuracy.
- Hardware fits the shortlist: IDSSE 2.6 GB, Metrica 183 MB, SkillCorner about 0.9 GB (extrapolated) against 300 GB of disk.

### Inferences
| Priority | Action | Effort | Risk |
|---|---|---|---:|
| 1 | Build the Metrica rehearsal set (ball-NaN dead, event owner) | Under 1 day | Noisy labels; 3 matches |
| 2 | Hybrid control labels on IDSSE with ELASTIC sync | 1–2 days | May overfit event taxonomy |
| 3 | Treat ball-NaN pattern as a dead-ball feature design question | 1 day | Automatic gaps differ from corrected |
| 4 | State-conditional `degrade()` | 2–3 days | Over-concentrating gaps on dead ball |
| 5 | Train on IDSSE + Metrica samples | 1 day | Metrica licence ambiguity |
| 6 | Calibrated prior EM and bounded decoder rates, if rehearsal supports them | 1–2 days | Silent drift on target |
| 7 | One round of conservative self-training, if rehearsal supports it | 1–2 days | Confirmation bias |
| 8 | SkillCorner as extra validation or H/A pretraining | 2 days | 10 fps; inferred dead state |
| — | PFF | Unknown | Blocked until terms are read |

Effort figures are judgement, not sourced.

### Gaps
- None of these changes has been tested for spellF1 yet. The expected gain of each is unknown.
- Whether organisers accept Metrica sample data with no named licence, or PFF data, needs a direct answer from them.
