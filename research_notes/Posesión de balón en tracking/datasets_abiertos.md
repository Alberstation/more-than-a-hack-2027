# Open football datasets combining tracking with possession / ball-in-play / events (for a per-frame H/A/D model)

Context: we need to train and validate a model that labels each 25 fps frame as H (home possession, ball alive), A (away possession, ball alive) or D (dead ball). It runs on automatic, uncorrected Metrica tracking (0–1 normalised coordinates, ball missing in about 41% of frames). Notes current as of 2026-10-28. Items marked "(unverified)" come from prior knowledge, not from a page fetched in this session.

## Q1. Which open datasets have per-frame ball alive/dead AND ball-owning team? (priority catalogue)

### Takeaway
Only one open dataset has both labels as native per-frame fields at 25 Hz: IDSSE / DFL Bundesliga (7 matches, TRACAB, CC-BY 4.0). It is the main training and validation set. SkillCorner open data (10 A-League matches, 10 fps) has a per-frame possession team, but its ball-in-play status has to be derived. Metrica's own sample data (3 matches, 25 fps) is closest to the target in format and provider. Its labels have to be derived from events, which include team, start/end frames and set-piece/out events.

### Cited Findings

**1. IDSSE / DFL Bundesliga open data (Bassek et al. 2025): top priority**
- Paper: Bassek, Rein, Weber, Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12:195 (2025), DOI 10.1038/s41597-025-04505-y — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- 7 full matches from the 2022/23 season. Two are Bundesliga: 1. FC Köln–FC Bayern (1:2, 2023-05-27) and VfL Bochum–Bayer Leverkusen (3:0, 2023-05-27). Five are 2. Bundesliga, all with Fortuna Düsseldorf at home, against Nürnberg, Regensburg, St. Pauli, Rostock and Kaiserslautern. 10 teams, 207 players — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- Tracking: TRACAB Gen5 optical, 16–20 cameras, **25 Hz**, players **and ball** (x, y, plus ball Z) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- **Per-frame `BallPossession`** (1 = home, 2 = away, based on the player in control) and **per-frame `BallStatus`** (0 = inactive, 1 = active; inactive when the referee interrupts play, e.g. fouls or substitutions) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- Events: 11,137 in total, in a hierarchy of player/team/referee actions (Play, ShotAtGoal, Foul, TacklingGame, set pieces, substitutions, whistles, ...) with ISO 8601 timestamps — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- Format: 3 XML files per match (match info, events, positions; DFL/Sportec format) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- Event and position clocks are separate processes. A DataBallPy sync of 6,251 passes and shots gave a mean shift of −0.37 ± 1.82 s (max 27.34 s). Known issues: occlusion in crowded situations such as corners, ID swaps, and event timestamp errors — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- License **CC-BY 4.0** (with DFL authorisation). Figshare DOI 10.6084/m9.figshare.28196177. Also loadable with *floodlight* — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/); [HF mirror README](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- kloppy: `sportec.load_open_tracking_data(match_id="J03WPY")` and `sportec.load_open_event_data(...)`. Match IDs: J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WPF, J03WQF. A Hugging Face mirror is at pysport/idsse-data — [HF README](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- The kloppy loader takes `sample_rate`, `limit`, `coordinates` and `only_alive`. **`only_alive` defaults to True, so dead-ball frames are dropped unless you pass `only_alive=False`. This matters for learning D.** J03WMX is Köln vs Bayern — [search summary of kloppy/unravelsports docs](https://unravelsports.readthedocs.io/en/latest/api/soccer/dataset.html)
- (Unverified) kloppy maps BallStatus to `frame.ball_state` (alive/dead) and BallPossession to `frame.ball_owning_team` for Sportec.

**2. Metrica Sports sample-data: priority 2 (same provider and format family as the target)**
- 3 anonymised games. Games 1–2 are CSV, one tracking file per team. Game 3 (added 2021-04-15) uses EPTS FIFA format for tracking and metadata (XML metadata + TXT tracking) and JSON for events. Tracking and events are synchronised. Event types are documented in a PDF in `documentation/` — [GitHub](https://github.com/metrica-sports/sample-data); [Sample_Game_3 folder](https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_3)
- License: none named. The README asks users to "be responsible with the use of this data" and to credit the source if used publicly — [GitHub](https://github.com/metrica-sports/sample-data)
- The README does not mention a per-frame ball status or possession field — [GitHub](https://github.com/metrica-sports/sample-data)
- kloppy: `metrica.load_open_data(match_id=1, limit=1000)` — [kloppy docs](https://kloppy.pysport.org/user-guide/concepts/tracking-data/)
- (Unverified, from familiarity with the files) 25 fps. Coordinates are normalised 0–1, as in the target. Ball x/y are NaN when the ball is not tracked. Events have Team, Type (PASS, BALL LOST, RECOVERY, SET PIECE, BALL OUT, CHALLENGE, ...), Start/End Frame and 0–1 coordinates. Possession and dead-ball intervals can be reconstructed from team + start/end frames, from BALL OUT and SET PIECE events, and from the frames in between.

**3. SkillCorner open data: priority 3 (per-frame possession team, lower fps, broadcast)**
- **10 matches** from the 2024/25 Australian A-League (the repo was refreshed; older versions had 9 European matches). Tracking at **10 fps**. Body-pose files for 2 matches at 25 fps, ~3.3 GB each on Hugging Face — [GitHub](https://github.com/SkillCorner/opendata)
- Ball included (`ball_data` per frame). **Per-frame `possession` = {player_id, group (home/away)}** — [GitHub](https://github.com/SkillCorner/opendata)
- Also includes dynamic events CSV (`player_possession` rows with EPV and pressure), a phases-of-play CSV with start/end frames, and `{id}_match.json` / `{id}_tracking_extrapolated.jsonl` — [GitHub](https://github.com/SkillCorner/opendata)
- License: the repo says MIT. SkillCorner asks for credit and notification on Twitter. No separate data license — [GitHub](https://github.com/SkillCorner/opendata)
- Quality: about 97% of player identities are correct, and speed/acceleration need smoothing — [GitHub](https://github.com/SkillCorner/opendata)
- (Unverified) kloppy has `skillcorner.load_open_data(...)`. In older versions, possession group `null` was often used for ball-not-in-play or uncontested phases. There is no explicit alive/dead flag, so check whether phases_of_play marks dead-ball phases.

**4. PFF FC FIFA World Cup 2022 (now under the "Gradient Sports" brand): priority 4**
- 64 matches. Broadcast tracking per match (`{game_id}.jsonl.bz2`), all events in a single `events.json`, plus metadata/players/rosters/competitions CSVs, PDF specs, and PFF grades. Access is through a request form — [PFF blog](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data); [Gradient Sports mirror of the post](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data)
- The tracking is collected through PFF's partnership with Sportlogiq — [fc.pff.com](https://fc.pff.com/)
- Frame rate reported as ~30 Hz (secondary source) — [third-party write-up](https://stranger9977.github.io/wc2026-tracking-transformer/methodology.html)
- Loaders: fast-forward has a GradientSports/PFF JSONL loader — [fast-forward docs](https://fast-forward.readthedocs.io/en/latest/providers/gradientsports/). kloppy lists PFF as a provider — [kloppy](https://kloppy.pysport.org/)
- License: no license text found. The announcement only asks users to tag @PFF_FC — [PFF blog](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)

**5. SoccerTrack v2: video/GSR, not ready-made tracking with possession**
- 10 full university/amateur matches (~900 min) of 4K panoramic BePro video. GSR annotations give per-frame 2D pitch coordinates, jersey IDs, roles and teams. Ball Action Spotting has 12 classes (Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Block, Tackle, Free Kick, Goal) taken from BePro event logs — [arXiv 2508.01802](https://arxiv.org/html/2508.01802v1); [project page](https://atomscott.github.io/SoccerTrack-v2/)
- License: data **CC BY 4.0** and code MIT. The page says both permit academic and commercial use. Hosted on Hugging Face — [project page](https://atomscott.github.io/SoccerTrack-v2/)
- The page does not say whether ball coordinates are in the GSR files — [project page](https://atomscott.github.io/SoccerTrack-v2/)

**6. SoccerNet (Tracking / GSR / Action Spotting / Ball Action Spotting): low priority**
- SoccerNet-GSR is open, with data and code at github.com/SoccerNet/sn-gamestate. SoccerNet-Tracking has bounding boxes for players, referees and ball. Clips are short (30 s) broadcast clips — [arXiv 2404.11335](https://arxiv.org/pdf/2404.11335)
- No dataset license text was found. The CC BY on arXiv covers only the paper — [arXiv 2404.11335](https://arxiv.org/pdf/2404.11335)

**7. StatsBomb open data incl. 360: low priority (no continuous tracking)**
- Events plus 360 freeze-frames "for selected matches" in `data/three-sixty`, one file per match. Attribution required ("state the data source as StatsBomb and use our logo"), with terms in LICENSE.pdf — [GitHub](https://github.com/statsbomb/open-data)

**8. Alfheim / Tromsø (Pettersen et al., MMSys 2014): not useful**
- ZXY wearable tracking at 20 Hz, **players only**, and video. Matches include TIL vs Strømsgodset (2013-11-03) and vs Tottenham (2013-11-28). Re-identifying players or building player/club performance profiles is not allowed. About 100 GB — [Simula](https://datasets.simula.no/alfheim/); [Qualinet](https://qualinet.github.io/databases/video/soccer_video_and_player_position_dataset)

### Inferences
- **Training/validation plan:** use the 7 IDSSE matches (~7 × 90 min × 60 × 25 ≈ 945k frames) for supervised H/A/D labels. Native `BallStatus` gives D, and `BallPossession` × alive gives H/A. That is the label definition the hackathon uses for the target (corrected Metrica: status alive/dead + owning team).
- **Make IDSSE look like the target:** normalise coordinates to 0–1. Remove ball positions in about 41% of frames, in realistic bursts (dead-ball periods, crowded areas, aerial balls) rather than at random. Add jitter and ID swaps. This reproduces the "automatic, uncorrected" input.
- **Metrica samples:** use as a format sanity check and a cross-provider validation set. Labels are approximated from events, so treat them as noisy.
- **SkillCorner:** resample 10 to 25 fps. Its possession labels can pretrain the H/A head. It also adds broadcast-style gaps, where off-camera players are extrapolated.
- **PFF WC22:** 64 matches is the largest pool. Possession and dead-ball labels would have to come from events (possession events + timestamps). Confirm the license before use.
- **Licensing:** none of the datasets found explicitly forbids competition use. CC-BY 4.0 (IDSSE, SoccerTrack v2) is cleanest. Simula/Alfheim restricts performance profiling. StatsBomb, PFF and Metrica rely on attribution notes or user agreements, so the organisers should be told which datasets and terms were used.

### Gaps
- I could not open kloppy's provider pages, so I could not confirm in the docs that the Sportec loader fills `ball_state` / `ball_owning_team` per frame, or what the Metrica and SkillCorner loaders produce for those fields. Verify locally with `dataset.to_df()` and check the `ball_state` and `ball_owning_team` columns.
- PFF tracking schema (ball, possession or ball-status fields), exact fps and license terms: not confirmed from the PDF specs.
- StatsBomb LICENSE.pdf (non-commercial user agreement?) and the list of 360 competitions: not read.
- Metrica sample frame rate and event taxonomy: not confirmed from the files in this session (the README states neither).
- I did not research Last Row / Friends of Tracking (I believe 19 Liverpool goal sequences with players and ball, but no possession or status labels), Signality, Second Spectrum or NWSL samples. TeamTrack was also not checked. Treat all of these as unassessed.
