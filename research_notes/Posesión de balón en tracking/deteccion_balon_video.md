# Ball detection and re-tracking in football video (to fill ball-missing frames)

Context: a fixed wide-angle tactical camera (1920x1080, 25 fps, ~199k frames), per-frame homography available, provided tracking missing the ball on ~41% of frames. Constraints: the method must be algorithmic and reproducible, with no hand labels on this match and no cloud APIs. Notes compiled 2026-10-28.

## 1. Which ball detectors and trackers exist, and what are their soccer-specific metrics?

### Takeaway
The best-documented and best-licensed option is **WASB-SBDT** (NTT, BMVC 2023, MIT, pretrained weights). On a 1920x1080, 25 fps fixed-camera soccer dataset it reports F1 88.3 / AP 83.6 at about 56 FPS (Step=3). Its setup matches this match's camera almost exactly. In the same benchmark, TrackNetV2 reached F1 86.6. Classic single-frame ball detectors (DeepBall, BallSeg) scored far lower, around F1 36-45. The quickest way to get a working detector is a YOLO or RF-DETR model with tiling (SAHI-style slicing), but per-frame detectors still miss many balls ("frequent false negatives"). They also need a temporal tracker on top.

### Cited Findings
**WASB-SBDT (top priority)**
- Repo: https://github.com/nttcom/WASB-SBDT, the official code for the BMVC 2023 paper "Widely Applicable Strong Baseline for Sports Ball Detection and Tracking" (Tarashima et al.). It covers soccer, tennis, badminton, volleyball and basketball. — [GitHub](https://github.com/nttcom/WASB-SBDT); [arXiv 2311.05237](https://arxiv.org/abs/2311.05237)
- License is MIT. Pretrained WASB weights are listed in MODEL_ZOO.md. — [GitHub](https://github.com/nttcom/WASB-SBDT)
- Repo dates: arXiv 11/09/2023, released 11/17/2023 with WASB evaluation code, TrackNetV2/ResTrackNetV2/MonoTrack evaluation code added 11/21/2023, DeepBall/DeepBall-Large/BallSeg added 11/23/2023. Tested on Python 3.8, CUDA 11.3, Ubuntu 18.04. A Dockerfile is recommended. — [GitHub](https://github.com/nttcom/WASB-SBDT)
- Soccer dataset: the D'Orazio et al. (2009) ISSIA-style set, six synchronized fixed cameras at **1920x1080, 25 FPS**. It has 17,993 frames (11,994 train from 4 clips, 5,999 test from 2 clips). The authors **re-annotated** the ball because the original labels "do not localize ball position correctly." — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)
- Soccer results at distance threshold τ = 4 px, reported as F1 / Acc / AP / FPS: DeepBall 44.5/92.7/26.3/44.6; DeepBall-Large 44.9/89.5/34.0/42.0; BallSeg 36.1/92.6/20.0/64.8; TrackNetV2 86.6/97.7/77.2/66.0; ResTrackNetV2 84.6/97.4/75.5/56.2; MonoTrack 85.2/97.4/78.6/58.0; **WASB Step=3 88.3/97.9/83.6/55.7**; WASB Step=1 88.2/97.9/86.2/23.6. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)
- Input: frames resized to 288x512, with 3 consecutive frames stacked on the channel axis (a multi-frame heatmap network in the TrackNet family). Experiments ran on a server with 4x V100 GPUs. The paper does not say which GPU produced the FPS figures. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)
- Online tracker: candidates are taken from the heatmap, a position is predicted from a local motion model (position + velocity + acceleration), candidates too far from the prediction are dropped, and the highest-confidence remaining candidate is kept. The authors found **no gain from Kalman or particle filters**. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)
- Limitations: at most one ball per frame. Only validated at HD/FHD and about 25-30 fps. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)

**TrackNet family**
- TrackNetV3 (qaz812345/TrackNetV3) adds "InpaintNet" trajectory rectification, which repairs missing segments by inpainting the predicted trajectory. Checkpoints are TrackNet_best.pt + InpaintNet_best.pt (TrackNetV3_ckpts.zip). It was designed for badminton shuttlecocks. The license is reported as MIT by derivatives and mirrors, but the upstream LICENSE file was not verified. Last push was reported as Apr 15, 2025. — [HF derivative card](https://huggingface.co/deadfast/beach-volley-vision-models); [Gitee mirror](https://gitee.com/tslnihao/TrackNetV3); [gittrend](https://gittrend.io/repo/qaz812345/TrackNetV3)
- TrackNetV4 (Raj, Wang, Gedeon; arXiv 2409.14543, ICASSP 2025) adds motion-attention maps from frame differencing as a plug-in to V2/V3. It was evaluated only on tennis and shuttlecock data. I found no confirmed official code repo. — [arXiv](https://arxiv.org/abs/2409.14543v1); [project page](https://tracknetv4.github.io)
- An independent HackMD reproduction found TrackNetV4 + attention performed *worse* than TrackNetV2 + attention, so the gains are not settled. — [HackMD](https://hackmd.io/@XDU5dRI5RJOkP6SWYyGUQA/HkQA5OZaJg)

**Integrated player and ball detectors (older)**
- FootAndBall (Komorowski et al., VISAPP 2020; https://github.com/jac99/FootAndBall) is MIT. It is a fully convolutional network with a ball confidence map and player boxes, built for long-shot high-resolution soccer video, with about 100x fewer parameters than SSD/YOLO. Its pretrained weights were trained on ISSIA-CNR cameras 1-4 + SoccerPlayerDetection set 1. — [GitHub](https://github.com/jac99/footandball); [arXiv 1912.05445](https://arxiv.org/pdf/1912.05445)
- FootAndBall notes the ball spans only about 8-20 px in long shots and is easily confused with body parts or litter on the pitch. — [arXiv 1912.05445](https://arxiv.org/pdf/1912.05445)
- DeepBall, from the same group, is a ball-only detector evaluated on ISSIA-CNR. It is clearly beaten by temporal heatmap models in the WASB benchmark (F1 44.5 vs 88.3). — [SciTePress](https://scitepress.org/Papers/2019/73489/); [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)

**General-purpose detectors fine-tuned on the ball (Roboflow)**
- roboflow/sports (https://github.com/roboflow/sports) is MIT. It ships Roboflow Universe datasets: soccer ball detection `football-ball-detection-rejhg`, player detection `football-players-detection-3zvbc`, and pitch keypoints `football-field-detection-f07vi`. It lists ball tracking as an open challenge. — [GitHub](https://github.com/roboflow/sports)
- Roboflow's ball tutorial (Aug 6, 2024) used YOLOv8x for 50 epochs on DFL Bundesliga Data Shootout frames sampled at 3 fps, at imgsz 640 with 2x2 tiles. It reports mAP@0.5 0.925 and mAP@0.5:0.95 0.57. Inference used `sv.InferenceSlicer` (slice = half frame + 100 px, overlap 100 px, NMS IoU 0.1) at conf 0.3. — [Roboflow blog](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- The tutorial's tracker keeps a deque of the last 10 ball centres and picks the detection nearest their centroid. The authors admit "frequent mistakes, especially false negatives". Listed failure modes are small size, motion blur, crowd backgrounds for aerial balls, heads and equipment as look-alikes, and glare. — [Roboflow blog](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- A newer Roboflow pipeline (Jul 2026) uses two RF-DETR models + BoT-SORT and keeps only the single highest-confidence ball per frame. — [Roboflow highlight blog](https://blog.roboflow.com/make-automatic-highlight-reels-from-soccer-games/)
- RF-DETR code and COCO-pretrained weights are **Apache 2.0**. — [PyPI rfdetr](https://pypi.org/project/rfdetr/); [Roboflow blog](https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/)

**SoccerNet**
- SoccerNet-Tracking (CVPRW 2022) has 200 sequences of 30 s plus one full 45-min half, with boxes and IDs for players, referees **and the ball** from broadcast camera views. The authors stress that ball tracking is "far from being solved". A secondary page cites 297 ball tracklets / 215,156 ball boxes (not verified against the paper). — [arXiv 2204.06918](https://arxiv.org/pdf/2204.06918); [alphaxiv benchmark](https://www.alphaxiv.org/benchmarks/kaust/soccernet-tracking)
- sn-gamestate (GSR baseline, https://github.com/SoccerNet/sn-gamestate) is **GPL-3.0** and uses YOLOv11 (Ultralytics) for detection plus TVCalib/PnLCalib/NBJW calibration. Its scope is **athletes only, not the ball**. The latest TrackLab fix was 2026-05-01. — [GitHub](https://github.com/SoccerNet/sn-gamestate)
- SoccerNet 2025 has no ball-detection challenge. Its tasks are GSR, Team Ball Action Spotting, and others. — [arXiv 2508.19182](https://arxiv.org/abs/2508.19182)

### Inferences
- **The recommended baseline is WASB.** Its soccer benchmark uses the same resolution, frame rate and kind of fixed full-pitch camera as the hackathon match. It is MIT-licensed, runs locally, and its temporal 3-frame heatmap is the type of model that beat per-frame detectors by more than 40 F1 points.
- The pretrained soccer weights come from a 2009-era dataset (other stadium, lighting and camera). Expect a domain gap. Fine-tuning on public labelled data (SoccerNet-Tracking ball boxes converted to centre points, the Roboflow ball dataset, ISSIA) is allowed and is likely needed.
- A self-training loop is a legal, reproducible way to adapt to this match without hand labels. Steps: (1) use the provided ball tracking on the ~59% of frames where it exists as pseudo-labels (projected back to the image through the inverse homography), (2) fine-tune WASB or YOLO on those frames, and (3) infer on the missing frames. This is an inference, not a published recipe. It should be confirmed against the hackathon rules, since the labels come from the provided data rather than from hand labelling.
- Time budget: at about 56 FPS (WASB Step=3, V100-class), 199k frames take about 1 h. Step=1 (23.6 FPS) takes about 2.3 h. A consumer RTX 3060/4070 is probably in the same range. These are estimates, not measured. Running only on gap frames ±N frames cuts the cost by about half.
- YOLO/RF-DETR with 2x2 or 3x3 tiling at 1080p means 4-9 forward passes per frame, about 0.8-1.8M passes for the match. That is feasible overnight on one GPU, but slower than WASB.

### Gaps
- The WASB paper does not state the GPU used for the FPS figures. I found no measured consumer-GPU throughput.
- I found no published ball-class HOTA/DetA from the SoccerNet-Tracking paper (the PDF could not be parsed).
- I found no peer-reviewed benchmark comparing YOLO/RF-DETR + slicing against WASB on the same soccer data.
- I could not confirm whether TrackNetV4 has official code.

## 2. How are small, occluded and aerial balls handled, and what about 3D height and trajectory interpolation?

### Takeaway
Multi-frame heatmap models (TrackNet/WASB) and tiling help with small balls. Occlusion gaps are filled by trajectory inpainting (TrackNetV3 InpaintNet), motion-model gating (WASB), or learned inference of the ball from player positions (Set Transformer + Bi-LSTM, KDD 2023). Monocular 3D ball height is possible with camera calibration but is still research-grade. The best real-soccer evidence is about 26 cm height RMSE on a tiny dataset.

### Cited Findings
- WASB's gating uses a constant-acceleration local motion model. Kalman and particle filters gave no improvement on their benchmarks. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)
- TrackNetV3's InpaintNet repairs masked or missing trajectory segments by inpainting. — [Gitee mirror README](https://gitee.com/tslnihao/TrackNetV3)
- "Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM" (KDD 2023, arXiv 2306.08206) infers the ball trajectory from player trajectories. It first predicts the possessing player, then adds a reality loss and post-processing for physically plausible paths. It is proposed explicitly for filling gaps a video detector misses. No code link was found, and the metrics are not in the abstract. — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206)
- arXiv 2602.12080 (2026) models possession as a sequence of edges on a dynamic player graph. It notes that methods without physical constraints yield implausible ball paths and too many possession changes. — [arXiv 2602.12080](https://www.arxiv.org/pdf/2602.12080)
- Kontos & Karlis (2023, Italian J. Applied Statistics) compare interpolation methods for gaps in football tracking data. — [IJAS](https://oajournals.fupress.net/index.php/ijas/article/view/15707)
- SportsLabKit fills ball-track gaps by interpolation. — [SportsLabKit docs](https://sportslabkit.readthedocs.io/en/latest/notebooks/02_user_guide/05_tracking_the_ball.html)
- "Where Is The Ball" (arXiv 2506.05763, Jun 2025) uses an LSTM to lift 2D monocular ball tracks to 3D using camera intrinsics and extrinsics. It is trained on simulation. On Real IPL soccer (9 sequences, triangulated ground truth) it reports NRMSE 0.74 vs 14.33 (Mocanu) and 5.03 (Shen), with RMSE 119 cm distance and 26 cm height. It requires the first and last points on the ground, ignores spin and aerodynamics, and fails on chest touches. Code is "will be released" (where-is-the-ball.github.io). — [arXiv HTML 2506.05763](https://arxiv.org/html/2506.05763v1)
- A CVPRW 2026 (CVsports) paper benchmarks physics models for 3D ball trajectories from monocular soccer video. Only its supplemental PDF was found. — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/supplemental/Grad_Physics-Based_3D_Ball_CVPRW_2026_supplemental.pdf)
- The classic physics approach (Kim et al., ICCV 1998) fits parabolic flight using the ground-to-image transform and estimates height relative to player height. — [mlanthology](https://mlanthology.org/iccv/1998/kim1998iccv-physics)

### Inferences
- The per-frame homography maps the image to the ground plane, so it is only valid while the ball is on the ground. During aerial balls the projected pitch position is biased away from the camera. For possession, this matters less than detecting *that* the ball is airborne. A parabola fit in image space between two ground contacts (Kim 1998 style) is a cheap way to flag aerial segments and to interpolate them.
- Recommended gap filling: (a) a short gap (<~10-15 frames) between two confident detections gets linear or quadratic interpolation. (b) For a longer gap, assign the ball to the nearest player of the team in possession before and after, if it is the same team, or run a learned model in the style of arXiv 2306.08206. (c) A dead-ball (D) gap can be detected via ball outside the pitch plus player-configuration cues.

### Gaps
- I found no public, licensed code for 3D soccer ball height from a single tactical camera that is ready to use.
- I found no reported metrics for arXiv 2306.08206 (only the abstract was read).

## 3. Licenses, weights and throughput (including AGPL for Ultralytics)

### Takeaway
To avoid AGPL, prefer MIT/Apache tools: WASB (MIT), FootAndBall (MIT), TrackNetV3 (reportedly MIT), roboflow/sports + supervision (MIT), RF-DETR (Apache 2.0). Ultralytics YOLOv8/YOLO11 code **and the weights trained with it** are AGPL-3.0 by default. That is fine for an open-source hackathon repo, but problematic for anything kept proprietary.

### Cited Findings
- Ultralytics says all YOLO-trained models fall under AGPL-3.0 by default, even when trained from scratch or used internally. Compliance means open-sourcing the whole project under AGPL or buying an Enterprise License. Staff say private on-prem delivery does not satisfy AGPL. — [Ultralytics GitHub yolo11](https://github.com/ultralytics/yolo11); [Ultralytics forum](https://community.ultralytics.com/t/yolov11-agpl-license-for-on-premises-use/1529)
- RF-DETR code and COCO weights are Apache 2.0. — [PyPI](https://pypi.org/project/rfdetr/)
- sn-gamestate is GPL-3.0 and depends on Ultralytics YOLOv11. — [GitHub](https://github.com/SoccerNet/sn-gamestate)
- WASB is MIT with pretrained weights. — [GitHub](https://github.com/nttcom/WASB-SBDT)
- FootAndBall is MIT with pretrained weights. — [GitHub](https://github.com/jac99/footandball)
- roboflow/sports is MIT, with datasets on Roboflow Universe. — [GitHub](https://github.com/roboflow/sports)
- Speeds: WASB 55.7 FPS (Step=3), TrackNetV2 66 FPS, BallSeg 64.8 FPS, all at 288x512 on a V100-class server. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)

### Inferences
- All candidates run fully offline, which satisfies the "no cloud upload" rule. The Roboflow hosted inference API must **not** be used. Download the datasets and weights and run locally.
- If the hackathon repo will be public, AGPL is not blocking, but this should be stated in the README.

### Gaps
- I found no Enterprise License pricing for Ultralytics.
- The TrackNetV3 upstream LICENSE file was not directly verified.

## 4. Open soccer datasets for fine-tuning a ball detector

### Takeaway
Usable public ball labels exist in SoccerNet-Tracking (broadcast, ball boxes), the Roboflow football ball dataset (DFL Bundesliga frames), ISSIA-CNR / D'Orazio (fixed 1080p cameras, re-annotated by WASB), and SoccerTrack (fisheye/panoramic). SoccerTrack v2's per-frame ball boxes were only partly released, through a challenge subset. ISSIA and SoccerTrack are the closest to a wide-angle tactical view.

### Cited Findings
- ISSIA / D'Orazio 2009: six fixed cameras at 1920x1080, 25 fps, about 18k frames, with ball re-annotated by the WASB authors. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)
- SoccerNet-Tracking: 200 x 30 s broadcast clips + a 45-min half, with ball boxes. — [arXiv 2204.06918](https://arxiv.org/pdf/2204.06918)
- Roboflow `football-ball-detection-rejhg`: built from DFL Bundesliga Data Shootout video frames. — [GitHub roboflow/sports](https://github.com/roboflow/sports); [Roboflow blog](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- SoccerTrack v2 (arXiv 2508.01802): 10 university matches, 932 min of fixed **4K panoramic** video, MOT/GSR + ball action spotting labels in 12 classes. Full ball/player box annotations for 1.62M frames were planned, but only a curated subset was released for the SoccerTrack Challenge (MMSports 2025). — [arXiv 2508.01802](https://arxiv.org/html/2508.01802v1)
- The SoccerTrack Challenge 2025 was scored on player HOTA. The winner, GTATrack (Deep-EIoU + global tracklet association), reached HOTA 0.60. No ball-specific scores were found. — [arXiv 2602.00484](https://arxiv.org/abs/2602.00484); [challenge site](https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025)
- SoccerTrack v1 (CVPRW 2022): a fish-eye wide-angle soccer dataset and tracker. — [CVF](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf)
- TeamTrack: full-pitch multi-sport MOT (soccer, basketball, handball). Whether it has a ball class was not confirmed. — [arXiv 2404.13868](https://arxiv.org/abs/2404.13868v1)

### Inferences
- For a 1080p tactical camera, combine ISSIA (same geometry), the Roboflow/DFL ball set and SoccerNet-Tracking ball boxes. Add pseudo-labels from the provided tracking on this match, after checking the rules.

### Gaps
- Licenses of the individual datasets (ISSIA, DFL Kaggle, SoccerNet NDA, SoccerTrack) were not verified in this pass. SoccerNet requires signing an NDA for its videos.
- I did not verify whether TeamTrack or SoccerTrack v2 public splits include per-frame ball positions.

## 5. What is proven to work best on wide-angle tactical footage?

### Takeaway
On the only public benchmark matching this setup (fixed 1080p/25 fps full-pitch cameras), **temporal heatmap models (WASB > TrackNetV2 ≈ MonoTrack) clearly beat single-frame detectors**. Recommended priority: (1) WASB, fine-tuned or self-trained on this match, (2) gap filling with motion and physics interpolation plus player-context possession inference, (3) a tiled RF-DETR/YOLO detector as a second opinion or ensemble.

### Cited Findings
- WASB F1 88.3 vs DeepBall 44.5 / BallSeg 36.1 on 1080p fixed-camera soccer. — [arXiv HTML 2311.05237](https://arxiv.org/html/2311.05237)
- YOLOv8x + slicing reaches mAP@0.5 0.925 on broadcast DFL frames, but leaves frequent false negatives in video. — [Roboflow blog](https://roboflow.com/blog/tracking-ball-sports-computer-vision)
- Ball tracking in SoccerNet broadcast data is "far from being solved". — [arXiv 2204.06918](https://arxiv.org/pdf/2204.06918)

### Inferences
- The provided tracking misses the ball most in crowded boxes, aerial balls and tackles. These are exactly where occlusion defeats any detector, so re-detection alone will not reach 100%. A practical goal is to cut the 41% missing rate substantially and resolve the rest at the possession level, from player context plus team continuity, rather than by exact ball position.

### Gaps
- I found no published head-to-head on a modern elite-stadium tactical camera, so expected recovery rates on this match are unknown and must be measured on held-out frames where the provided ball exists.
