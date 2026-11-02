# Additional football tracking and event datasets for the possession model

Research date: 2026-11-01. Delivery: 17/11/2026, so 16 days remain. Scope: enlarge training beyond 7 IDSSE matches, with 3 Metrica matches as rehearsal. This extends `research_notes/Posesión en tracking cinco mejoras/datos_y_dominio.md`.

**[measured]** means queried directly in this session from a GitHub, Hugging Face or Figshare API, or computed from a raw file. No tracking dataset was downloaded. Nothing about the challenge match was opened.

**Corrections to earlier notes:**
- SkillCorner now has **20** matches, not 10.
- SoccerTrack v2 does not have to be rejected for its video size if its annotation folders can be fetched alone. That selective download is not yet confirmed.

## Which datasets exist, and how do they rank for this task?

### Takeaway
Only three sources add full matches of usable player tracking with a possession signal in the time available:
- **SkillCorner (20 matches):** ready today, MIT.
- **PFF/Gradient World Cup 2022 (64 matches):** largest and contains Spain, but its terms remain unread.
- **SoccerTrack v2 (10 amateur matches):** CC BY 4.0. Possession would be built from its ball-action events.

Download SkillCorner now, submit the PFF form today, and also refit on the 3 Metrica matches at the end. Everything else is rejected or serves only as a prior.

### Cited Findings

**Ranked inventory** (rank is judgement, see Inferences)

| # | Dataset | Matches, Hz, level | Ball | Possession label | Licence | Access | Size |
|---|---|---|---|---|---|---|---|
| 1 | SkillCorner Open Data | **20**, A-League 2024/25 [measured, [matches.json](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json)]; 10 fps, broadcast — [README](https://github.com/SkillCorner/opendata) | Yes, partly extrapolated | Per-frame `possession.group` — [README](https://github.com/SkillCorner/opendata) | MIT — [LICENSE](https://raw.githubusercontent.com/SkillCorner/opendata/master/LICENSE) | Open, Git LFS | One tracking file 90.7 MB [measured], so about 1.8 GB for 20 (extrapolated) |
| 2 | PFF/Gradient WC 2022 | 64, about 29.97 Hz, broadcast — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data); [arXiv 2506.06666](https://arxiv.org/pdf/2506.06666) | Yes, 3D — [SoccerNet-GAR card](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) | Event-driven `home_ball` — [kloppy](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py) | **Unread** | Form on release page | Not found |
| 3 | SoccerTrack v2 | 10 university matches, 934 min — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) | **Reconstruction only**, interpolated between events — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) | Derive from 21,428 ball-action events with acting team — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) | CC BY 4.0, cite paper — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) | HF `gated: auto`: share HF info, accept conditions [measured, [API](https://huggingface.co/api/datasets/atomscott/soccertrack-v2)] | 145 GB per card; 256.9 GB `usedStorage` [measured]. Annotation folders' size unknown |
| 4 | Metrica sample | 3, 25 Hz per prior notes, anonymised | Yes | Derive from events | None named: "Please be responsible with the use of this data"; "If you use it for anything public, please aknowledge the source" — [README](https://github.com/metrica-sports/sample-data) | Open | About 183 MB |
| 5 | StatsBomb 360 | Freeze frames at events, not tracking. La Liga 2020/21: 35 matches, all Barcelona, all with 360 [measured, [matches file](https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/11/90.json)] | At event | Event possession sequences | User agreement, quoted below | Public repo; registration requested | Not measured |
| 6 | Wyscout (Pappalardo) | Events only, top leagues 2017/18 — [Sci Data](https://www.nature.com/articles/s41597-019-0247-7) | No tracking | Event order | CC BY 4.0 — [Figshare API](https://api.figshare.com/v2/articles/7770599) | Open | `events.zip` 77 MB [measured] |

**Rejected or not available**

| Dataset | Why |
|---|---|
| SoccerNet-GAR | **Not new matches:** the same 64 WC22 games, "constructed from the PFF FC website". 87,939 event windows, not continuous; full windows are 144 frames. 622 GB in the repo [measured]. Form: "research and non-commercial use only" and "not to share the dataset with anyone else" — [HF card](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR/blob/main/README.md) |
| SoccerNet GSR | 200 clips of 30 s, 2.36M athlete positions — [arXiv](https://arxiv.org/abs/2404.11335v1). Ball not listed in annotations — [repo](https://github.com/SoccerNet/sn-gamestate). 35 GB, ungated, GPL-3.0 tag [measured, [API](https://huggingface.co/api/datasets/SoccerNet/SN-GSR-2025)]. Clips are too short for possession spells |
| SoccerNet Tracking | 12 Swiss Super League games, but released as 30 s clips with 2D boxes — [task page](https://www.soccer-net.org/tasks/tracking). 23.6 GB [measured]. No possession labels, no pitch coordinates stated |
| SN-PCBAS-2026, SN-BAS | Video, NDA, 479 GB for PCBAS [measured]. No tracking |
| Last Row | 19 goal sequences, unlicensed — [open-football](https://github.com/withqwerty/open-football) |
| Alfheim | 3 Tromsø matches, 20 Hz. Reported bans on re-identifying players and creating performance profiles — [Simula, via search](https://datasets.simula.no/alfheim/). The page failed TLS in this session, so these terms are second-hand |
| SoccerMon | Women's GPS, wellness and load data, CC BY 4.0 — [open-football](https://github.com/withqwerty/open-football). No synchronised ball or events stated |
| TeamTrack | 2 soccer recordings, no ball mention, dataset licence unstated — prior notes |
| Impect | Bundesliga 2023/24 events only, own terms in LICENSE.pdf, not read — [README](https://github.com/ImpectAPI/open-data) |
| Second Spectrum, Signality, HawkEye, TRACAB | kloppy loaders exist, but no public data link is listed for them — [kloppy docs table](https://raw.githubusercontent.com/PySport/kloppy/master/docs/user-guide/loading-data/index.md) |
| IDSSE/DFL, more matches | Still 7. No new open DFL release found — [HF mirror](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md). In-house 3D tracking since 2025-26 is not an open release — [SVG](https://svgplay.sportsvideo.org/videos/3d-player-tracking-in-the-bundesliga) |
| Kaggle WC 2026 "match-state and ball-position" set | Derived from a paid provider, non-commercial, per search summary — [Kaggle](https://www.kaggle.com/datasets/jodach/fifa-world-cup-2026-dataset). Not tracking of all players as far as seen. Avoid |

**SkillCorner update**
- The commit of 2026-09-08 reads "Add 10 A-League matches and refresh six to dynamic events v3" [measured, [commits](https://api.github.com/repos/SkillCorner/opendata/commits?per_page=8)]. The README still says 10 matches, so it is stale. `matches.json` and the tree both show 20 [measured].
- Body pose for 2 matches was added on 2026-09-09 [measured]. It is not useful here.
- Reported use: the SkillCorner x PySport Analytics Cup ran on this data — [SkillCorner](https://skillcorner.com/event-2026/5-02-skillcorner-x-pysport-analytics-cup). A 2027 edition page exists — [SkillCorner](https://skillcorner.com/analytics-cup-2027).

**PFF reported use**
- Several papers use it, all on passes or defending, none on possession prediction: [2506.06666](https://arxiv.org/pdf/2506.06666), [2603.28916](https://arxiv.org/pdf/2603.28916), [2609.24040](https://arxiv.org/pdf/2609.24040), [2606.19931](https://arxiv.org/pdf/2606.19931).
- A paper's use is not evidence of what the terms allow.

**No newer release:** no 2025-2026 open tracking dataset was found beyond the SkillCorner extension and SoccerTrack v2 — [open-football](https://github.com/withqwerty/open-football). That list also says open data from Africa, Asia and South America barely exists.

### Inferences
- **Why this order.** SkillCorner ranks first for certainty: licence clear, download today, automatic broadcast tracking with a real missing pattern, and it takes the training pool from 7 to 27 matches. PFF has the largest upside. If its terms permit use, it moves to first.
- **Metrica in final training.** After rehearsal choices are frozen, refit on 7 + 3 matches. This costs nothing and adds 43% more matches from the target provider. The price is no rehearsal score for that final model, so run leave-one-Metrica-out beforehand to confirm that adding them helps. The missing licence is a question for the organisers.
- **SoccerTrack v2 caution.** Its `ball/` track is built from the same events that would supply the labels. A model given that track could learn the label from the ball's kinks. Hide it or use players only. If used, treat it as the last option because the level is amateur.
- **Marginal value is untested.** Before committing effort, run a learning curve: train on 2, 4 and 6 IDSSE matches and score on Metrica rehearsal. A rising curve justifies PFF work. The +0.011 to +0.018 from extra draws does not prove that new matches give the same. It only suggests variance is a limiter.

### Gaps
- Frame rate of SoccerTrack v2 annotations was not stated on the pages read.
- PFF size was not found. The tracking subfolder sizes were not inspected, and the root Drive listing shows none.
- Exact Metrica frame rate here comes from prior notes, not rechecked.

## What is the licence and access status, and how long does approval take?

### Takeaway
No provider publishes an approval time. With 16 days left, only instant or self-service routes are realistic. The competition rule matters as much as the licence: NDA data that cannot be shared with organisers is unsafe, so SoccerNet's gated sets are out.

### Cited Findings

**Competition rules**
- "The organisers can access and inspect it. Anything in your pipeline has to be something we can look at and run — your repository, a public model, a commercial API, an open dataset." — [RULES.md](../../RULES.md)
- "External data about other games" is allowed — [RULES.md](../../RULES.md)
- "A private in-house model or an unavailable internal dataset is the one thing this rules out. If your approach depends on something you cannot share, ask before you build on it." — [RULES.md](../../RULES.md)

**Licence clauses, primary sources**

| Dataset | Clause | Competition reading |
|---|---|---|
| SkillCorner | "Permission is hereby granted, free of charge" (MIT, © 2020) — [LICENSE](https://raw.githubusercontent.com/SkillCorner/opendata/master/LICENSE). "we kindly ask that you credit SkillCorner" — [README](https://github.com/SkillCorner/opendata) | Allowed. Credit it |
| IDSSE | CC BY 4.0 — [Figshare API](https://api.figshare.com/v2/articles/28196177) | Allowed. Credit it |
| SoccerTrack v2 | CC BY 4.0, accept HF conditions — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) | Allowed under the licence. Read the gate conditions when accepting |
| Wyscout | CC BY 4.0 — [Figshare API](https://api.figshare.com/v2/articles/7770599) | Allowed. Cite the papers |
| Metrica | No licence; responsible use and acknowledgement requested — [README](https://github.com/metrica-sports/sample-data) | Ambiguous. Do not redistribute; ask organisers |
| StatsBomb | "The User may not: 1.2.1. edit, distort, distribute, reproduce, sell or in any way provide the data to any external or third party; 1.2.2. commercially exploit the data or any analysis derived from the use of the Service" — [agreement](https://raw.githubusercontent.com/statsbomb/open-data/master/LICENSE.pdf) | Research tool, but "commercially exploit" and "edit" are broad. Keep out of the repo, ask if used |
| SoccerNet NDA | "Use only for non-commercial research purposes and do not distribute to tiers" — [NDA Exhibit A](https://drive.google.com/file/d/1Efz8yP-baa8CtcMB7SYIID7Bn5sH7sGTB4-u1YdVRf4/view). Text fetched through the document's text export | Conflicts with organiser inspection unless they sign too |
| PFF/Gradient | **Not found.** Release pages state none — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data); [enhanced post](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset). The [terms page](https://www.gradientsports.com/terms-conditions) and its [archived copy](https://web.archive.org/web/2025/https://www.gradientsports.com/terms-conditions) contain, in the captured text, only a 2022 API *Sandbox* agreement, which is a different service | Blocked until read |

**Access routes**

| Route | Datasets | Time |
|---|---|---|
| None | SkillCorner, Metrica, IDSSE, Wyscout, SN-GSR, SN-Tracking 2023 | Immediate |
| Self-acceptance | SoccerTrack v2 (`gated: auto` [measured]) | Should be immediate. No source gives a time |
| Web form | PFF: "simply fill out the form below and follow the easy-to-use instructions" — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data) | Not stated |
| Manual plus NDA | SoccerNet-GAR, PCBAS, BAS, raw video: "Access is granted once the request is approved" — [SoccerNet](https://www.soccer-net.org/data) | Not stated. NDA has personalised fields, and its counterparty is KAUST — [NDA](https://drive.google.com/file/d/1Efz8yP-baa8CtcMB7SYIID7Bn5sH7sGTB4-u1YdVRf4/view) |
| Sales or academic | SkillCorner full catalogue, StatsBomb tracking | No application route found. SkillCorner is described as a paid product — [Henshaw 2026](https://www.liamhenshaw.com/writing/where-to-find-football-data) |

**PFF access note.** kloppy's docs link a Google Drive folder directly — [kloppy docs](https://raw.githubusercontent.com/PySport/kloppy/master/docs/user-guide/loading-data/index.md). Its root lists Event Data, Metadata, Rosters and Tracking Data, plus a change log shared on 3 Jun 2025 and no licence file — [Drive listing](https://drive.google.com/drive/u/0/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa). I looked at the root listing only and downloaded nothing. The fc.pff.com site returned HTTP 530 today, so the form on gradientsports.com is the route to try. Use the form rather than the direct link, because the form is where terms would be accepted.

### Inferences
- **No approval-dependent plan should carry the delivery.** Send the PFF form today. If terms are not visible within a few days, drop it.
- **A prize competition may or may not be "commercial".** That is a legal question, and the sources do not settle it. Ask the organisers one combined question: may we train on Metrica samples and on PFF data under its terms, given that you can obtain them yourselves?
- **Enrolment terms outrank everything.** RULES.md says they are the authority for match assets and does not restate them. They were not available here.
- All these datasets are usable without uploading challenge data. None asks for it. Using them adds no human labels on the challenge match.

### Gaps
- PFF terms text is the single most important unknown. No approval time for it was found.
- Whether SoccerNet would grant access for a hackathon is unknown.
- Whether a form-gated free dataset counts as "open" under RULES.md is an organiser call.
- Impect's LICENSE.pdf and the Simula terms were not read.

## Which datasets contain non-Bundesliga, Spanish or high-possession teams?

### Takeaway
No open full-tracking La Liga or Barcelona dataset exists. PFF is the only route to Spain with tracking. StatsBomb has 35 Barcelona matches with freeze frames, useful at most for statistics such as spell-length priors.

### Cited Findings
- No open synchronised La Liga tracking was found. Published La Liga work pairs StatsBomb events with licensed SkillCorner tracking at 10 fps — [arXiv 2505.14711](https://arxiv.org/pdf/2505.14711); [arXiv 2506.16696](https://arxiv.org/pdf/2506.16696).
- PFF covers all 64 WC22 matches, so Spain is included — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data). Spain held about 81-82% against Costa Rica, depending on the source: 81.8% per [The Analyst](https://theanalyst.com/2022/11/spain-vs-costa-rica-world-cup-stats/), 81% per [FBref](https://fbref.com/en/matchs/dadb1883/Spain-Costa-Rica-23-Novembre-2022-FIFA-World-Cup).
- StatsBomb competitions with 360 include La Liga 2020/21, 1. Bundesliga 2023/24, Euro 2020 and 2024, Ligue 1, MLS 2023, AFCON 2023, WC 2022, and women's Euros and WWC 2023 [measured, [competitions.json](https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json)]. La Liga 2020/21 matches are all Barcelona (35 of 35) [measured].
- League mix available in tracking: A-League, German top two divisions, anonymous Metrica, national teams (PFF), Japanese-context university football (SoccerTrack) — sources in first table.

### Inferences
- **Weight rather than filter.** With PFF, up-weight possession-dominant matches. Choose them from the data's own possession share, not from team names, and do not fit a single-team model.
- **StatsBomb as a prior only.** Barça spell lengths from its possession ids could sanity-check decoder rates, subject to the licence question. Its definition differs from the challenge's, so the prior must be loose, and it must not be tuned to the public excerpt.
- Football style is a smaller risk than provider and label definition, given the earlier finding that cross-provider F1 falls sharply. That is judgement, not measurement.

### Gaps
- How many Spain matches PFF has and which Barça players were in that squad were not checked with a source.
- SoccerTrack's teams and country were not read directly from its card. The "Japanese" description above is an inference from the authors, so verify it.

## How are events or provider fields turned into labels compatible with BallPossession?

### Takeaway
Every source needs its own mapping, because "possession" differs by provider. Tools for this are permissive (kloppy BSD-3, socceraction MIT, databallpy MIT), and the recipe is simple: dead from the provider, last on-ball team otherwise, then the control filter from earlier notes.

### Cited Findings

| Tool | Licence | Role |
|---|---|---|
| kloppy | BSD-3-Clause [measured, GitHub API](https://api.github.com/repos/PySport/kloppy) | Loads SkillCorner, Sportec, Metrica and PFF into one tracking model — [docs](https://kloppy.pysport.org/user-guide/loading-data/) |
| databallpy | MIT [measured, GitHub API](https://api.github.com/repos/Alek050/databallpy) | `add_team_possession`: per-frame home or away from synchronised events — [docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html) |
| socceraction | MIT [measured, GitHub API](https://api.github.com/repos/ML-KULeuven/socceraction) | Event conversion. Not a frame labeller, so of little use here |
| ELASTIC | MPL-2.0 — prior notes | Sync for matches where events are not synchronised |
| unravelsports | Not checked | Infers owning team from distance thresholds and warns it may be inaccurate in contested possession — [docs](https://unravelsports.readthedocs.io/en/latest/api/soccer/dataset.html) |

- databallpy's own example has event-based and provider possession agree on 5,298 of 5,892 alive frames, or 89.9% (my calculation from its table). Its docs cite sync errors, provider flag errors and definition differences as causes — [docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html).

**Mapping per dataset**

| Dataset | H/A | D | Risk |
|---|---|---|---|
| SkillCorner | `possession.group` | kloppy marks ALIVE only when group is a team — [kloppy](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/skillcorner.py). So dead is inferred | Null on 50.6% of frames in one match; mean spell 8.9 s — prior notes. Null may mean contested, not dead. `phases_of_play` can cross-check |
| PFF | Last event with non-null `home_ball`, held until the next | DEAD only on `BALL_OUT`, `PERIOD_END`; ALIVE on kick-offs and `ON_THE_BALL` — [kloppy](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py) | Fouls and stoppages likely stay alive under this rule. Ball is `ballsSmoothed`, so its error pattern differs from raw automatic output |
| SoccerTrack v2 | Acting team of last controlling action | `Out`, then `Throw In`, `Free Kick` or goal restart gives dead intervals | `Ball Player Block` and `Tackle` need rules. Own rule, unvalidated |
| Metrica | Events, already in repo | Ball NaN — prior notes | Ball NaN does not carry over to degraded data |
| StatsBomb | Not frame level | Not frame level | Not applicable |

### Inferences
- **Train H/A and D with separate masks.** Use SkillCorner frames for the team head only where `group` is set, and do not count null as D until `phases_of_play` shows what null means. That protects the D class from a wrong assumption. This fits a two-stage or two-head design, and with HistGradientBoosting it means two models or a mask by sample weight.
- **Apply the same control filter to all.** Earlier notes found raw IDSSE flicker. Mixing flickery SkillCorner labels with cleaner ones will mostly teach short spells. Merge or filter consistently, with rules fixed on non-challenge data, then compare spell statistics across sources.
- Use a source indicator only for analysis, not as a feature, since the challenge's source value would be undefined.
- Degrade all new data with care. SkillCorner is already automatic, so applying the full `degrade()` would double the noise, and PFF is smoothed broadcast tracking. Use lighter or no degradation for these, and test by Metrica rehearsal.
- Frame rates differ, so features using windows or speeds must be built in seconds, not frames. Resample everything to one rate first. SkillCorner at 10 fps implies about 5 Hz after sampling, which suits `TRAIN_STRIDE = 5` at 25 Hz.

### Gaps
- No published comparison of `possession.group` or `home_ball` against a human definition was found.
- PFF `home_ball`'s precise meaning needs the spec PDF, which is delivered with the data. Measured spell lengths are unavailable until then.
- Floodlight's repository licence could not be confirmed because the repository path tried was wrong.

## What are the memory costs, and how should the data be loaded on 15.7 GB?

### Takeaway
Memory is manageable if each match is converted once to a small cached file and draws are reduced for new sources. The risk is PFF with the current settings of 3 draws and 2 gap modes, which could reach several GB of float32 features.

### Cited Findings
- `train_v3.py` uses `DRAWS = 3`, `GAPS = ("uniform", "dead")`, and `TRAIN_STRIDE = 5`; each match is taken from both sides — [train_v3.py](../../possession/train_v3.py); [train.py](../../possession/train.py).
- PFF ships as one compressed `.jsonl.bz2` per match — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data).

### Inferences
Estimates, assuming 140,000 frames per match at 25 Hz and F features of 4 bytes:

| Setup | Rows per match | Memory per match | 64 matches |
|---|---|---:|---:|
| Current: 3 draws × 2 gaps × 2 sides, stride 5 | 336,000 | 1.34 MB × F | about 86 MB × F |
| New source: 1 draw × 1 gap × 2 sides, stride 5 | 56,000 | 0.22 MB × F | about 14 MB × F |

- With F = 100 (an illustrative number, not checked against the code), that is 8.6 GB against 1.4 GB. Use the second setup for PFF.
- SkillCorner at 10 fps with stride 2 has similar row counts, about 2.7 GB of float32 at F = 100 under current settings for 20 matches.
- Parse with a streaming reader, keep only needed fields, and save `float32` npz per match. Do not load whole JSONL files together.
- HistGradientBoosting bins features into small integers internally, so fit-time memory is lower than the raw arrays, but the raw concatenation before fit is the peak to control.

### Gaps
- Actual feature count and peak RAM were not measured.
- PFF decompression size is unknown.

## Are other-sport datasets useful for pretraining?

### Takeaway
No. The current model cannot be pretrained, and the licences are poor. Skip them before 17/11.

### Cited Findings
- NBA SportVU: `linouk23/NBA-Player-Movements` holds 636 `.7z` logs, 3.8 GB in total [measured], and has **no licence** [measured, [API](https://api.github.com/repos/linouk23/NBA-Player-Movements)]. A mirror states that 2015-16 was the last public season — [HF](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16/raw/main/README.md). `sealneaward/nba-movement-data` is tagged MIT [measured], but that tag cannot grant rights over NBA data. This is my reading.
- Hockey: the Big Data Cup provides competition data, and its repos have no licence [measured, [2025 repo](https://api.github.com/repos/bigdatacup/Big-Data-Cup-2025)].
- NFL Big Data Bowl: data sits on Kaggle competition pages — [Kaggle](https://kaggle.com/competitions/nfl-big-data-bowl-2025/overview). The rules page could not be read.

### Inferences
- HistGradientBoosting has no pretraining mechanism. Other-sport data could only be added as rows, and pitch geometry, team size and rule differences make those rows poorly matched.
- It would be worth revisiting only for a neural sequence model with a shared encoder, which the deadline makes unlikely.
- Absence of a licence does not grant permission, so these datasets also add compliance risk for the organisers' inspection.

### Gaps
- No study was found that shows cross-sport pretraining helping possession inference in football.

## What should be done, in order?

### Takeaway
Start with SkillCorner and a learning curve. Keep a PFF option open with one form and one organiser question.

### Cited Findings
- Rule: anything in the pipeline must be inspectable by the organisers — [RULES.md](../../RULES.md).
- Prior result: PathCRF's authors attribute a higher within-dataset score to a larger training set, and cross-dataset F1 fell strongly — [PathCRF](https://arxiv.org/pdf/2602.12080).

### Inferences
| Order | Action | Effort | Risk |
|---|---|---|---:|
| 1 | Today: submit PFF form; send organisers the combined question | Minutes | None |
| 2 | Learning curve on IDSSE, scored on Metrica rehearsal | Hours | Few rehearsal matches |
| 3 | Download SkillCorner 20. Cache to the repo's format outside git | Half a day | Label semantics |
| 4 | Add SkillCorner to the team head with masks; compare on Metrica | 1 day | Domain mismatch |
| 5 | If PFF terms permit, add 1 draw per match with mask rules; validate on Metrica | 2-3 days | Time; smoothed ball |
| 6 | Try SoccerTrack v2 annotations only if 4-5 show matches help | 1-2 days | Amateur level; label leakage from `ball/` |
| 7 | Final refit including Metrica, after leave-one-Metrica-out check | Hours | Licence ambiguity |

Effort is judgement. Accept a source only if it improves Metrica rehearsal spellF1 across all 3 matches or their mean by a margin chosen beforehand. Do not use the public excerpt for this choice.

### Gaps
- None of the additions has been tested for spellF1, so expected gains remain unknown.
- Whether selective HF download of `gsr/` and `bas/` works for SoccerTrack v2 is unconfirmed. Its card does not say.
