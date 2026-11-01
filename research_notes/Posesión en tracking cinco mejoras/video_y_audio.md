# Video and audio signals for team possession and dead-ball inference (excluding ball detection)

Compiled 2026-11-01, deadline 17 Nov 2026 (16 days). Local checks were read-only, ran on this repo, and export only aggregates. "Inferences" are my reasoning, not sourced facts.

**Three corrections to the brief, found in the repo:**
1. **The tactical camera is not fixed.** Sampling 1 in 250 homography rows, `h00` runs 1218-4329 and `h12` 72-670. At the median frame the pitch mid-line ends project to x = -354 and x = 2311, i.e. wider than the 1920 px frame. So it pans and zooms. This helps transfer (see question 6). — local measurement on [BAR-ATH_tactical_homography.csv](data/tactical/BAR-ATH_tactical_homography.csv)
2. **A broadcast video of the same match is an allowed input.** It is on its own unsynchronised timeline and has its own tracking and homography. It is the native domain of all these models and may carry better audio. — [DATA.md](DATA.md); [FAQ.md](FAQ.md)
3. **The MP4s are not on this machine.** `data/broadcast/` does not exist and `data/tactical/` holds no `.mp4`. Neither do `~/Downloads` or the other folders checked, so **the audio check could not be run and nothing below has been tested on the match video**.

Audio check, once the video is present (`ffprobe` 7.1 is installed):
`ffprobe -v error -select_streams a -show_entries stream=codec_name,sample_rate,channels,duration -of csv=p=0 data/tactical/BAR-ATH_tactical.mp4` (repeat for the broadcast). Empty output means no audio track. A track could still be silent, so also run `ffmpeg -i <file> -af volumedetect -f null -`.

## SoccerNet action spotting state of the art: metrics, repos, licences, weights, speed

### Takeaway
Two models are deployable without training: **lRomul's 2023 Ball Action Spotting winner (MIT, weights public, pass/drive only, no team)** and the **T-DEED team baseline (GPL-3.0, weights public, 12 actions + left/right team)**. No RTX 4060 speed figure exists for any of them, and the top 2025 and 2026 systems publish no weights that I could find.

### Cited Findings
| Year / task | Winner | Score | Code, licence, weights |
|---|---|---|---|
| 2023 Action Spotting (17 classes) | SDU VSISLAB | tight avg-mAP 71.31 (ASTRA 3rd, 70.10) | Not checked for the winner; no ASTRA repo or licence found |
| 2023 Ball Action Spotting (Pass, Drive only) | Ruslan Baikulov | mAP@1 86.47 | [MIT, weights on Google Drive](https://github.com/lRomul/ball-action-spotting) |
| 2024 Ball Action Spotting (12 classes) | T-DEED | mAP@1 73.39 (baseline 56.15) | [GPL-3.0, Google Drive checkpoints](https://github.com/arturxe2/T-DEED) |
| 2025 Team Ball Action Spotting (12 classes × team) | dudek | Team-mAP@1 60.03, mAP@1 61.56 | [dude.k: no weights published, no licence file visible](https://github.com/int8/dude.k) |
| 2025 baseline | T-DEED + team head | Team-mAP@1 51.72, mAP@1 58.38 | [GPL-3.0, one Google Drive checkpoint](https://github.com/SoccerNet/sn-teamspotting) |
| 2026 Player-Centric Ball Action Spotting | FSI-Tahakom (PAVE) | Macro F1 58.94 (baseline 46.41) | [No code link in the results paper](https://arxiv.org/html/2607.07320) |

Sources for scores: [2023 results](https://arxiv.org/pdf/2309.06006), [2024 results](https://arxiv.org/pdf/2409.10587), [2025 results](https://arxiv.org/pdf/2508.19182), [2026 results](https://arxiv.org/html/2607.07320).

- The 2023 task has only Pass and Drive labels, 7 games and 11,041 timestamps, so 86.47 cannot be compared with 73.39 on 12 classes. — [2023 results](https://arxiv.org/pdf/2309.06006)
- lRomul input is 1280x736 grayscale at 12.5 fps, with EfficientNetV2-B0 "slow fusion" and flip TTA. Training was tuned on an RTX 3090 Ti, about 7 min 10 s per epoch. — [GitHub](https://github.com/lRomul/ball-action-spotting)
- The 2024 T-DEED ran 100-frame clips at 796x448. — [2024 results](https://arxiv.org/pdf/2409.10587)
- The dude.k winner used 224x224, 80-frame clips, and an RTX 4090. — [author's blog](https://int8.io/team-ball-action-spotting-challenge-2025)
- Retraining dude.k needs about 1.2 TB of disk, and extraction can take up to 24 h. — [GitHub](https://github.com/int8/dude.k)
- The 2026 Ball Action Anticipation top score is 24.08 mAP_avg, which is too weak to use as a signal. — [2026 results](https://arxiv.org/html/2607.07320)
- Speed reference: the 2022 DFL Kaggle event task had to finish test inference within 9 h on Kaggle servers, estimated at ≥16.67 fps. — [Fixstars slides](https://speakerdeck.com/player/37475620112b41ceb78d5de04503e229)

### Inferences
- **lRomul is the best touch-timing candidate**, because it has MIT code and keeps the highest pixel count. Input is roughly 0.67 of the original 1080p scale. It says *when*, not *which team*.
- T-DEED's 8 GB fit is likely for inference at 224p and uncertain at 448p. This is a guess that needs a smoke test.
- Retraining any of these is out of scope: it needs SoccerNet data access, far more disk, and a GPU above 8 GB.

### Gaps
- No published FPS or VRAM for T-DEED or lRomul on an RTX 4060, and T-DEED's paper has no efficiency table.
- SoccerNet video requires an NDA for some sets. I did not find terms governing weights trained on them.
- The separate "LICENSE E2E-Spot" file in T-DEED was not read.
- The 2024 Action Spotting winner was not extracted.

## Team-aware spotting: public models and accuracy

### Takeaway
Team-aware spotting exists, but **the T-DEED team baseline is the only one with public weights that I found**. No source publishes plain team accuracy. Its left/right output is direct evidence on the exact bottleneck, so it deserves a time-boxed test, but the numbers are from broadcast video and transfer is unproven.

### Cited Findings
- 2025 models must say which team, left or right relative to the camera, performs each of the 12 actions; AP is computed per action-team pair at a 1 s tolerance. — [2025 results](https://arxiv.org/pdf/2508.19182)
- Team vs plain mAP@1 gap: dudek 60.03 vs 61.56; Intellindust 56.78 vs 63.39; UniBW Munich 55.36 vs 68.92; baseline 51.72 vs 58.38. — [2025 results](https://arxiv.org/pdf/2508.19182)
- The winner replaced dual heads with one joint team-action head, pretrained on 500+ broadcast matches, then fine-tuned. — [2025 results](https://arxiv.org/pdf/2508.19182)
- Intellindust's "team alignment" re-assigned team labels for actions like Shot and Goal by looking back 4 s at prerequisite actions, which "greatly" improved mAP. — [2025 results](https://arxiv.org/pdf/2508.19182)
- FOOTPASS has 54 full matches, 102,992 events, and 8 classes, labelled (frame, team, jersey, class). Video is Full HD 25 fps broadcast. — [GitHub](https://github.com/JeremieOchin/FOOTPASS)
- FOOTPASS baselines: pure visual TAAD F1 35.9, TAAD+GNN 52.1, TAAD+DST 67.5 (at τ=0.15, ±12 frames). Without multi-agent reasoning, DST drops to 49.1. — [FOOTPASS paper](https://arxiv.org/pdf/2511.16183v1)
- Tackle is close to unsolved: F1 0.0 (FOOTPASS DST), 0.044 (ME-DST, overall micro F1 0.778), 0.123 (winning system on validation), 0.056 on the challenge set for another entrant. — [FOOTPASS](https://arxiv.org/pdf/2511.16183v1); [ME-DST](https://arxiv.org/pdf/2608.01696); [PAVE](https://arxiv.org/html/2606.28389); [Retraining report](https://arxiv.org/html/2606.09679)
- About 20% of FOOTPASS events have no bounding box, and one entrant used a 22 GB GPU memory limit. — [Retraining report](https://arxiv.org/html/2606.09679)
- FOOTPASS's licence is unclear: its page says CC BY-NC 4.0 for annotations and baselines, while the repo lists Apache-2.0. — [GitHub](https://github.com/JeremieOchin/FOOTPASS)

### Inferences
- **The team gap is an upper-level hint, not a team accuracy.** A gap of 1.5-13.5 mAP points suggests team assignment is learnable, but cannot be converted to a percentage. The brief's H/A error at touches can't be predicted from it.
- **Left/right maps to H/A cheaply**, using each half's attack direction, but it is defined relative to the camera, and for a model it means "the team attacking left", not jersey identity. Check whether the head reads direction of play. If so, it may be correlated with tracking features you already have and add less than hoped.
- **Duels are the worst class in every paper**, so no visual model is likely to fix contested touches, which are probably where the possession flips hide.
- FOOTPASS's most useful lesson is structural: most of the gain came from sequence reasoning over game state, not pixels. This supports a video-free stage trained on IDSSE, though that sits outside this note.
- FOOTPASS is unusable as a shortcut: no weights are stated, it needs SoccerNet video under NDA, and memory exceeds 8 GB. CC BY-NC would also be a flag.

### Gaps
- No team classification accuracy conditional on a detected action was found in any paper or repo.
- The checkpoint's training terms and exact T-DEED team head behaviour on a different camera are unknown.

## Player pose and player-ball interaction as a kick or receive detector

### Takeaway
Pose is cheap to run and licence-friendly but **has no evidence of working at this scale**, and one paper reports that pose alone failed to find the on-ball player. Treat it as a low-priority tiebreaker, not a primary signal.

### Cited Findings
- RTMPose-m: 75.8 AP on COCO, about 11.06 ms ONNX Runtime on an i7-11700, and about 2.29 ms TensorRT FP16 on a GTX 1660 Ti. rtmlib runs it on ONNX Runtime without PyTorch or MMCV. — [MMPose RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- Body8 weights are trained on AI Challenger, COCO, CrowdPose, MPII, sub-JHMDB, Halpe and PoseTrack18. — [MMPose RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- ViTPose is listed as Apache 2.0 by a third-party catalogue, with base AP 75.8. — [Voxel51](https://docs.voxel51.com/model_zoo/models/vitpose_base_torch.html)
- A 2025 skeleton-based paper reported that its pose-only model did not identify on-the-ball players on its own. This comes from a search snippet because the page returned 403. — [ResearchGate](https://www.researchgate.net/publication/397520715_Skeleton-based_Approach_for_On-the-ball_Player_Identification_on_Broadcast_Soccer_Videos)
- Kick detection precedent exists but used multi-view 4K video and a ball-centred window. — [ITE paper listing](https://www.researchgate.net/publication/340326656_Papers_Keep_Your_Eye_on_the_Ball_Detection_of_Kicking_Motions_in_Multi-view_4K_Soccer_Videos)
- PathCRF (2026) detects on-ball events from player trajectories alone, with no ball. — [arXiv 2602.12080](https://arxiv.org/pdf/2602.12080)
- Local measurement: pitch scale is about 21 px/m at the far touchline, 29 at mid-pitch, 45 at the near touchline (medians; p5-p95 overall 18-62). — [homography CSV](data/tactical/BAR-ATH_tactical_homography.csv)

### Inferences
- A 1.8 m player is about 32-110 px tall, so roughly 40 px on the far side and 80 on the near side. Ankle keypoints are plausible near and doubtful far. The ball is about 4-14 px. Foot-ball contact cannot be read from pose on the far side.
- Running all frames costs about 4.4M crops (22 × 199k). That is about 13 h on one CPU thread, or about 3 h on GPU, extrapolating the published latencies. Restricted to ±0.5 s around a few thousand ambiguous touches among 2-3 candidates, it falls to minutes, and CPU ONNX is the easiest route to determinism.
- Best use is a binary feature at contested touches: which candidate has a kicking-leg extension or a body lean towards the ball.

### Gaps
- The MMPose LICENSE and per-dataset weight terms were not verified in the fetched text. AI Challenger and Halpe terms may restrict use, so check them before shipping. A non-commercial flag is possible.
- ViTPose's licence is verified only through aggregators.
- No study was found for pose-based touch detection at a tactical scale.

## Team classification from jersey colour to fix tracking team errors

### Takeaway
This is the most certain pipeline to run, but **it likely fixes the wrong problem**. It corrects wrong team labels on tracks, while the oracle gap is about choosing which team touched the ball. Measure the tracking team error rate first.

### Cited Findings
- `roboflow/sports` is MIT licensed. — [GitHub](https://github.com/roboflow/sports)
- Its classifier uses `google/siglip-base-patch16-224`, `umap.UMAP(n_components=3)` and `KMeans(n_clusters=2)`, **with no `random_state` set**. — [team.py](https://raw.githubusercontent.com/roboflow/sports/main/sports/common/team.py)
- SigLIP base is Apache 2.0. — [Hugging Face](https://huggingface.co/google/siglip-base-patch16-224)
- Roboflow's tutorial crops the central part of the box and clusters once per stable track. — [Roboflow](https://blog.roboflow.com/identify-basketball-players/)
- A simpler alternative in third-party code is KMeans over HSV jersey histograms. — [sportvision](https://pypi.org/project/sportvision/0.2.2/)
- No football accuracy figure was published for this approach in any page found. — [Roboflow](https://blog.roboflow.com/identify-basketball-players/)

### Inferences
- With k=2, goalkeepers and referees get forced into a team. Use k=3-5, or exclude them by track role.
- For one match with known kits, a per-track-segment Lab colour histogram with a fixed seed is simpler, runs on CPU, and is deterministic. Prefer it over SigLIP unless it fails.
- Track identity swaps need per-segment rather than per-track votes.
- Unset seeds and UMAP's parallelism break byte-identical output. Fix `random_state`, and set `n_jobs=1` for UMAP.
- Value test: inject label noise at the rate you measure into the current pipeline (the `dose_exp.py` approach) before building anything.

### Gaps
- The current tracking team label error rate is unknown to me and is the deciding number.
- The team classifier needs video; it could not be tested here.

## Referee detection, whistle audio, and referee gestures for dead ball

### Takeaway
**There is no off-the-shelf validated referee whistle detector.** A transparent band-energy detector is the most practical route, and is deterministic if an audio track exists. Whistles mark only some dead-ball moments. Referee gesture recognition is a negative result.

### Cited Findings
- A 2011 bandpass-threshold method on about 12 h of mixed sports audio found 324 whistles, with 92.5% sensitivity, 80.5% reliability, 42 false positives and 26 false negatives. — [Amrita](https://www.amrita.edu/publication/automated-referee-whistle-sound-detection-for-extraction-of-highlights-from-sports-video/)
- A 2025 soccer paper uses 1 s windows with 50% overlap and targets 3300-4200 Hz with STFT features, Naive Bayes, SVM, neural nets and PSD post-processing. Its abstract says "highly effective" and gives no metrics. — [Trinh et al.](https://publications.isods.org/index.php/jdsai/article/view/33)
- A 2014 detector exploits a three-tone whistle structure and entropy-based rejection of non-tonal frames. — [arXiv 1411.6496](https://arxiv.org/pdf/1411.6496)
- bit-bots/whistle_detection is MIT, archived, has no stated accuracy or weights, and uses 10 RoboCup recordings. — [GitHub](https://github.com/bit-bots/whistle_detection)
- YAMNet predicts 521 AudioSet classes, expects 16 kHz mono audio, and outputs a "Whistle" label on a train-whistle example, which is not a referee benchmark. — [TensorFlow](https://www.tensorflow.org/hub/tutorials/yamnet); [MathWorks](https://www.mathworks.com/help/audio/ref/yamnetpreprocess.html)
- PANNs are MIT with Zenodo weights, Cnn14 mAP 0.431 on AudioSet, 32 kHz and 16 kHz variants, and decision-level frame-wise models. Whistle-class performance is not stated. — [GitHub](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- Audio improved soccer action spotting in earlier work, and ASTRA uses audio for non-visible actions. — [Vanderplaetse & Dupont](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w53/Vanderplaetse_Improved_Soccer_Action_Spotting_Using_Both_Audio_and_Video_Streams_CVPRW_2020_paper.pdf); [ASTRA](https://arxiv.org/abs/2404.01891)
- FRGR-YOLOv8s recognises six gestures on 1,200 still images, is frame-based, and lists dynamic gestures as future work. — [Frontiers](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10910025/)
- SoccerNet-Tracking labels include referees. — [arXiv 2204.06918](https://arxiv.org/pdf/2204.06918)
- The Action Spotting head of T-DEED covers the 17-class SoccerNet set. — [T-DEED repo](https://github.com/arturxe2/T-DEED); [2023 results](https://arxiv.org/pdf/2309.06006)

### Inferences
- **Use whistle events as a one-way cue.** Football rules and practice suggest whistles for fouls, offsides, goals, and restarts, and generally none for routine throw-ins and goal kicks. So they should raise dead-ball probability, and their absence means nothing.
- Combine a whistle with what tracking shows after it: a slowdown within a few seconds separates a real stoppage from crowd whistling or a neighbouring pitch sound. Fans' whistles are the main false-positive risk.
- A 50 m distance adds about 150 ms (4 frames) of sound delay, and a whistle follows the event it marks, so it is a dead-ball onset lag that needs calibration.
- Calibration can be automatic: pick band and threshold using half-time boundaries and Smart Tagging set-piece intervals, with no human labels. Smart Tagging is automatic and provided, per [DATA.md](DATA.md).
- Gesture recognition is not worth pursuing: no video dataset, no weights, and a far-side referee may be about 40 px tall.

### Gaps
- Whether either video has usable audio. Tactical feeds often carry none or only a stadium ambience track. This is the go/no-go check.
- No published precision for whistles in stadium audio with crowd noise was found after three searches.
- No public referee signal video dataset or ball-out-of-play audio study was found.
- The whistle rules claim above is general domain knowledge, not sourced here.

## Reality of transfer from broadcast-trained models to a tactical view

### Takeaway
Better than feared in one case, but the evidence is thin. On **unedited main-camera professional video, end-to-end models lost almost nothing on 17-class event spotting**. On auto-piloted cameras they dropped hard. No study covers ball-touch or team-head transfer to tactical video, so this must be tested on the match.

### Cited Findings
- Cabado et al. (CVPRW 2024) tested SoccerNet-trained models on Swiss Super League unedited main camera (9 matches), Luxembourg BGL AI-piloted, and amateur CINFO. — [paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf)
- End-to-end Tight-Average-mAP from Table 3 (row labels reconstructed from PDF column order, so verify in the PDF): SoccerNet 61-73, Swiss about 63-72, BGL about 22-41, CINFO about 19-26. — [paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf)
- Feature-based models collapsed: at most 5.08% tight on Swiss and below 1.10% on CINFO. — [paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf)
- The authors state end-to-end methods "exhibit excellent generalization in the Swiss Super League subset". — [paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf)
- Ball Action Spotting datasets come from English Football League broadcasts. — [2025 results](https://arxiv.org/pdf/2508.19182)

### Inferences
- The panning tactical view is closer to a main-camera broadcast than a static wide shot, which favours the Swiss-like outcome. It is still wider and higher, so expect some loss.
- The result covers mainly coarse, salient events with a 1-5 s tolerance, not sub-second touches. So it supports dead-ball events such as throw-in, corner and foul more than pass or receive.
- At 224p, a 40-110 px player becomes about 8-23 px and the ball about 1-3 px. Models at that size likely infer from movement and context, so a tighter pitch-following crop could help. Crop using the homography around the tracked players to match broadcast framing, as a low-cost test.
- With broadcast video available, the alternative is to run models natively on it, then map events to tactical time by aligning the two tracking data sets. It avoids transfer entirely, but only 65.8% of broadcast frames have a valid homography, plus 122 stretches of replays and close-ups per [DATA.md](DATA.md). Replays must be filtered out, and the alignment is unbuilt.

### Gaps
- No tactical-view evaluation of T-DEED, E2E-Spot or team heads was found.
- I could not confirm the camera type of the DFL Kaggle videos, or find its first-place write-up. That data is a possible close analogue.
- Validation has no labels on this match. Proxy checks are listed in the final question.

## What to try before 17 Nov, with effort, licences and determinism flags

### Takeaway
Run a **1-day go/no-go on audio and video transfer** before committing. Then the order is T-DEED team head on tactical, whistle band detector if audio exists, and lRomul touches, with pose and jersey colour only if their pre-checks justify it. Expect a modest, uncertain gain, and a realistic chance that none of these beats the tracking baseline.

### Cited Findings
- PyTorch does not promise reproducibility across releases, commits or platforms, and cuDNN can vary between runs; deterministic modes can slow execution. — [PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.md)
- Fused attention backends stay non-deterministic unless `warn_only=False` or the math backend is used. — [PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.md)
- NVIDIA says reproducibility is per hardware, and GPU model, CUDA or cuDNN version can change results. — [NVIDIA](https://docs.nvidia.com/tao/tao-toolkit/latest/text/deterministic_training.html)
- Submissions must have a command that regenerates the file, and a file the repo cannot regenerate is a rule problem. — [SUBMISSION.md](SUBMISSION.md); [RULES.md](RULES.md)
- No hosted LLM may receive the videos, and human-produced data about this game is not allowed. — [FAQ.md](FAQ.md)

### Inferences

| Priority | Signal | Effort | Licence flag | Determinism | Main risk |
|---|---|---|---|---|---|
| 0 | Audio probe on both videos | 10 min | none | n/a | No audio kills whistle work |
| 1 | T-DEED team head on tactical, crop test | 2-3 days | **GPL-3.0** code; SoccerNet-derived weights | GPU: flag | Left/right fails or just re-encodes attack direction |
| 2 | Whistle band detector plus tracking confirmation | 1-2 days | none (own DSP) | Deterministic on CPU if decode pinned | No audio, or crowd noise |
| 3 | lRomul Pass/Drive touch times | 1-2 days | MIT code; SoccerNet-derived weights | GPU: flag | Knows when, not which team |
| 4 | 17-class events for dead ball (T-DEED AS) | 1 day after item 1 | **GPL-3.0** | GPU: flag | Coarse timing |
| 5 | Pose tiebreaker at contested touches | 2-3 days | Verify MMPose and dataset terms | CPU ONNX likely stable | Far-side scale |
| 6 | Jersey team correction | 1 day | MIT and Apache, or own colour code | Set seeds | Not the real bottleneck |
| Later | Broadcast-to-tactical alignment | 3-5 days | same | same | Unbuilt, replays |

**Do not attempt:** retraining T-DEED or FOOTPASS, referee gesture recognition, or SigLIP-first team clustering. **Ask the organisers before using** commentary speech-to-text on the broadcast. It is a human description of this match, which may conflict with the ban on human-produced data.

- **Go/no-go gate:** take the first 10 minutes on the 4060 and record fps and VRAM.
- **Label-free validation:** compare team predictions to the team of the nearest tracked player at ball-speed kinks, the attack direction, and Smart Tagging set pieces. Agreement only with tracking means no new information, so the useful evidence is correct disagreement. That can only be measured with ball-tracked gaps held out, not proven.
- **Fusion:** use outputs as soft features at touches, keeping the current model as fallback, and require a dose test: if signal X were Y% accurate, what spellF1 would it give? Use oracle-noise injection to set the accuracy needed before building each one. A team signal must beat what tracking already gives at contested touches, not the 0.48 average.
- **Byte-identical regeneration:** GPU inference is the main risk. Pin decoder, driver, CUDA, cuDNN, and PyTorch; disable cuDNN benchmark; enable deterministic algorithms; and run single-process. Rounding scores to about 3 decimals before thresholding reduces drift but does not guarantee identity, since a value can land on a rounding edge. Test by running twice on the 4060 and diffing. The strongest option is CPU-only final regeneration, if runtime permits. Ship cached scores only if the rules allow it, because frame-level derived data may count as reconstructing data.
- **GPL:** the repo must be public anyway, so GPL-3.0 is workable for this challenge, but linking it could put the combined code under GPL. Keep it as a separate script producing a score file, and get the organisers' view. No GPL ban was found in [RULES.md](RULES.md) or [SUBMISSION.md](SUBMISSION.md), and the terms signed at enrolment were not read.
- **AGPL:** Ultralytics YOLO is not needed for any item above. Avoid it, including in gesture or roboflow examples. Licence from memory, not re-verified here.

### Gaps
- No published estimate of possession gain from any of these signals on any dataset. All benefit is unproven until tested.
- Effort estimates are mine, for one person, and assume the video and working CUDA are available.
- Enrolment terms on external weights and on GPL were not available to me.
