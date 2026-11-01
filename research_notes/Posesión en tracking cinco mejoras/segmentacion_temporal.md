# Temporal segmentation and structured decoding for possession spells (spellF1)

Evidence status, 2026-11-01. Numbers come from search-result extracts or fetched pages, not from a full reading of every paper. Items marked "extract" came from a search summary and should be checked before being quoted. No source measured spellF1 or football team-possession spell F1 directly, so all projections to this problem are inferences. The Jansche paper, the HSMM literature and the TCPD result tables could not be read in full (see Gaps).

## 1. Neural action segmentation: MS-TCN family, ASFormer, DiffAct, FACT, ASRF, BCN

### Takeaway
Temporal models gain segmental F1 even when frame accuracy barely moves, and explicit boundary refinement (ASRF) is the best-evidenced single addition (+8.6 to +16.1 F1@10 over MS-TCN). The heavier models (ASFormer, DiffAct, FACT) add smaller increments, and I found no evidence of them on tabular features with fewer than 10 sequences, so a small TCN with a boundary head is the only neural option worth testing before 17 Nov.

### Cited Findings

Published results, full-data settings (F1@10 / F1@25 / F1@50, then frame accuracy):

| Dataset (size) | Method | F1@10 | F1@25 | F1@50 | Acc | Source |
|---|---|---|---|---|---|---|
| 50Salads (50 videos) | Bi-LSTM | 62.6 | 58.3 | 47.0 | 55.7 | [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) |
| 50Salads | MS-TCN | 76.3 | 74.0 | 64.5 | 80.7 | [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) |
| 50Salads | ASRF | 84.9 | 83.5 | 77.3 | 84.5 | [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) |
| 50Salads | ASFormer | 85.1 | 83.4 | 76.0 | n/a | [EffiDiffAct, extract](https://arxiv.org/pdf/2408.02024) |
| 50Salads | DiffAct | 90.1 | 89.2 | 83.7 | n/a | [EffiDiffAct, extract](https://arxiv.org/pdf/2408.02024) |
| GTEA (28 videos) | MS-TCN | 85.8 | 83.4 | 69.8 | 76.3 | [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) |
| GTEA | ASRF | 89.4 | 87.8 | 79.8 | 77.3 | [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) |
| GTEA | FACT (third-party listing, not verified against paper) | 96.1 | 95.6 | 87.5 | 84.5 | [FACT search listing](https://hyper.ai/en/papers/fact-frame-action-cross-attention-temporal) |
| Breakfast | MS-TCN | 58.2 | 52.9 | 40.8 | 65.1 | [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) |
| Breakfast | ASRF | 74.3 | 68.9 | 56.1 | 67.6 | [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) |
| Breakfast | ASFormer | 76.0 | 70.6 | 57.4 | n/a | [EffiDiffAct, extract](https://arxiv.org/pdf/2408.02024) |
| Breakfast | DiffAct | 80.3 | 75.9 | 64.6 | n/a | [leaderboard extract](https://hyper.ai/en/sota/tasks/action-segmentation/benchmark/action-segmentation-on-breakfast-1) |
| Breakfast | FACT | 81.4 | 76.5 | 66.2 | n/a | [leaderboard extract](https://hyper.ai/en/sota/tasks/action-segmentation/benchmark/action-segmentation-on-breakfast-1) |

- ASRF splits segmentation into frame classification and class-agnostic boundary regression. At inference only, boundaries are local maxima above 0.5, the sequence is cut there, and each segment takes the majority class of the frame branch — [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866).
- ASRF over MS-TCN gains F1@50 +12.8 (50Salads), +10.0 (GTEA) and +15.3 (Breakfast), while accuracy gains only 3.8, 1.0 and 2.5 points (computed from the table above) — [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866).
- On IMU-derived skeleton features with 18 subjects (HuGaDB), MS-TCN and Bi-LSTM have near-equal accuracy (86.8 vs 86.1) but F1@50 is 89.9 vs 81.5; MS-GCN reaches 93.0. This is the closest non-video, low-dimensional evidence found — [search extract, HuGaDB table](https://arxiv.org/pdf/2202.01727).
- FACT claims to be 3 times faster than transformer methods, and its evaluation covers Breakfast, GTEA, EgoProceL and Epic-Kitchens, with no 50Salads benchmark. A mirror reports an MIT licence, unconfirmed in the repo — [FACT, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html); [supplement](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Lu_FACT_Frame-Action_Cross-Attention_CVPR_2024_supplemental.pdf); [repo](https://www.github.com/ZijiaLewisLu/CVPR2024-FACT); [mirror](https://awesome.ecosyste.ms/projects/github.com%2FZijiaLewisLu%2FCVPR2024-FACT).
- BCN uses boundary probabilities for local barrier pooling, and a 2026 preprint says its gains cost more than 16x the MACs and parameters of a simpler boundary-channel approach (secondary, unchecked) — [arXiv 2604.01859](https://arxiv.org/pdf/2604.01859).
- Football, 7 Bundesliga matches: sequence modelling was the main driver of gain over frame-based baselines in a T-GAN preprint (7-fold leave-one-match-out). A pure Transformer was within 0.4 to 2.2% frame F1 of the graph model at intention level. Post-processing helped but could not fully rescue weak raw predictions — [Li & Link 2026](https://arxiv.org/html/2606.09289v1).

### Inferences
- Size arithmetic: with about 437 spells per match, 7 matches give about 3,000 segments, more than the sequence count suggests. They are still correlated, and only 7 independent sequences exist for validation.
- The HuGaDB and ASRF patterns (flat accuracy, higher F1) match the problem here. Past a good frame classifier, the remaining gain is in boundaries and segment count, not frame probabilities.
- The gaps above come from raw-video features and action classes with long fixed scripts. Expect much less here because the 46 features already embed temporal context and H/A/D has no script. Treat these numbers as direction only, not as size of gain.
- Feasible design: downsample to 5 fps (about 32k steps), slide windows of a few minutes, take out-of-fold HGB log-probs plus a small feature subset, and use 1-2 stages, about 32-64 filters, with a frame head and a boundary head. An RTX 4060 is plenty. Out-of-fold inputs are mandatory, otherwise the TCN learns to trust overfit HGB outputs.
- Rule out DiffAct: its sampling adds a source of non-determinism, and it needs far more data and training than 16 days allow. ASFormer and FACT offer no clear benefit over a small TCN at this data size.

### Gaps
- No reduced-training-set result was found for ASRF, BCN, ASFormer, DiffAct or FACT. Their behaviour with 7 sequences is unknown, and I found no published failure either.
- No study of these models on tabular team-sport features.
- Licences of MS-TCN, ASFormer, DiffAct, ASRF and BCN repos were not verified.

## 2. Explicit-duration models, feature-dependent CRFs, change-point methods

### Takeaway
The strongest directly relevant evidence is PathCRF on the same class of data (7 DFL matches): structured decoding raised event F1 from 65.3 to 73.1, masking illegal transitions gave most of the rest, and learned feature-dependent transitions added about 1.9 points. That closely mirrors the 0.48 to 0.496 you saw. Pure change-point methods are unsupervised and should only serve as candidate generators.

### Cited Findings
- PathCRF, Sportec Open DFL dataset, 7 matches (5 train, 1 val, 1 test). Event F1: ball-trajectory postprocessing 48.39; non-CRF 65.34; non-CRF + Viterbi 73.14; Static Dense CRF 68.78; Dynamic Dense CRF 67.82; Static Masked CRF 73.83; Dynamic Masked CRF 75.69 — [PathCRF](https://arxiv.org/html/2602.12080v2).
- In PathCRF, edge accuracy barely moves between non-CRF and final model (69.24 vs 69.64) while event F1 gains about 10 points. Frame-level team possession accuracy was 92.40% — [PathCRF](https://arxiv.org/html/2602.12080v2).
- FIFA replaces exact duration-aware dynamic programming with gradient descent on a segment energy (length term plus observation term), more than 5 times faster with kept performance. A listing cites Breakfast F1 66.8 with MS-TCN, marked unverified. This shows length models are usable on segmentation outputs — [FIFA](https://arxiv.org/abs/2108.03894v1); [listing](https://sotaverified.org/papers/fifa-fast-inference-approximation-for-action).
- In ASRF's controlled comparison, boundary refinement was compared against relabelling, smoothing and similarity post-processing on the same frame output; the paper's overall claim is up to 16.1% segmental F1 gain — [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866).
- Change-point review: 37 series, 14 algorithms; binary segmentation and BOCPD are reported among the best. PELT's rank could not be retrieved — [van den Burg & Williams](https://arxiv.org/pdf/2003.06222).

### Inferences
- Dense learned transitions were worse than masked static ones in PathCRF (67.82 vs 73.83). With limited data, hard constraints beat free transition learning. For H/A/D, a learned transition should be a low-capacity function of a few features, for example boundary score, ball-speed change and dead-ball signals, with strong regularisation. This fits the mixed 5-up, 2-down outcome of the flip-cost experiment.
- HSMM duration risk: true duration distributions differ (mean 7 s IDSSE vs 14 s in the public sample). A duration prior fit on IDSSE would push the wrong segment count on the challenge match if that gap is real. Use weak or parametric durations, and test sensitivity to a 2x duration scale. A hazard-style duration term in a 3-state model can be implemented in NumPy without any library.
- Possession lengths are very likely heavy-tailed, with 30% under 2 s, and a geometric duration from constant switch cost underweights short and long spells at once. This is the theoretical case for HSMM, but it is untested here.
- Use `ruptures` only on signals that truly jump (the change-detector score or log-odds), as extra candidate boundaries for the decoder. PELT with one penalty repeats the constant switch-cost trade-off, so expect little independent gain over Viterbi. BOCPD's online nature brings nothing because the full match is available.

### Gaps
- I found no head-to-head of HSMM vs HMM on segmental F1 in action segmentation or sport, so no quantified gain is available.
- TCPD per-method tables could not be read because the PDF was not parsed. Default vs oracle sensitivity remains unreported here.

## 3. Decoders that optimise F1 or IoU instead of MAP

### Takeaway
Maximum expected F-score decoding is established and polynomial for restricted models, and MBR decoding has recently been applied directly to span F1, but I found no ready method for the IoU-matched, multi-threshold spellF1. A sampled-posterior MBR decoder is my best proposal: it needs no new training and targets the under-segmentation directly.

### Cited Findings
- Jansche (ACL 2007): maximum expected F-score decoding is polynomial under model assumptions, with hypotheses to evaluate linear in sequence length for a simple class of models — [Jansche 2007, search extract](https://preview.aclanthology.org/moar-dois/P07-1093.pdf).
- Jansche, minimum-risk chunking: an algorithm to efficiently evaluate expected F-measure under IOB2 chunking — [Algorithms for Minimum Risk Chunking](https://core.ac.uk/download/161444453.pdf).
- Posterior marginal decoding is the MBR special case for Hamming cost — [lecture slides](https://structuredprediction11763.github.io/structuredprediction.github.io/new_slides/7mar.mbr.pdf).
- MBR was introduced for n-best reranking (Kumar & Byrne 2004) and extended to lattices (Tromble et al. 2008) — [MBR survey page](https://www2.statmt.org/survey/Topic/MinimumBayesRisk).
- 2025 work applies MBR to error-span detection with span-level F1 as utility, and reports a defect in that metric on empty annotations — [arXiv 2512.07540](https://arxiv.org/pdf/2512.07540v1). A unifying MBR view of generation methods is in [arXiv 2310.01387](https://arxiv.org/pdf/2310.01387).

### Inferences
- Count bound: with 290 predictions against 437 truths, even if every prediction matched, F1 = 2·290/(290+437) = 0.80. That is the ceiling at each IoU threshold under the current count.
- Marginal rule (own derivation): F = 2TP/(|P|+|T|). Adding one prediction with match probability p raises expected F only if p > F/2. At F around 0.5, a new short spell is worth inserting at p > 0.25. This is a much lower bar than MAP implies, so MAP under-segments for this metric. The rule ignores that splitting a spell can break its neighbours, so check it by simulation.
- Short-spell geometry: a spell of length L with shift d in one boundary has IoU (L−d)/(L+d). For L = 2 s, IoU 0.5 needs d under 0.67 s and IoU 0.7 needs d under 0.35 s. So those 130 or so spells under 2 s pay only with sub-second boundaries, and boundary precision limits the gain more than detection does.
- Proposed decoder, all CPU and NumPy:
  1. Calibrate per-frame log-probs with a temperature. Frame autocorrelation makes HGB overconfident, so tune this out-of-fold.
  2. Draw M paths, about 200-500, by forward-filtering backward-sampling from the HMM or CRF lattice, using a fixed seed.
  3. Generate candidates: Viterbi across a grid of switch and dead costs, plus candidates from boundary thresholds, plus each sample.
  4. Score each candidate as mean spellF1 against all sampled paths and choose the best (MBR with the true metric).
  5. Optional local search: greedily insert, delete or shift boundaries.
  Candidates × samples cost is small because spell lists are only hundreds long.
- Simplest form first: tune cost and temperature per fold directly on spellF1. This is a coarse version of the same idea and tests whether the count shift pays.
- Do not fix a target count. The 7 s vs 14 s difference suggests the challenge match may have fewer spells, and the MBR decoder adapts only as far as the model posterior is calibrated on that match.
- Note on the noise level: 0.48 vs 0.496 with 5 up and 2 down across 7 folds is not distinguishable from noise on its own, so require paired per-match comparison and report the fold spread for every candidate.

### Gaps
- I found no published decoder for one-to-one IoU-matched F1 averaged over thresholds. Dynamic programming for exact expected spellF1 appears intractable because the matching is global; I have no proof or source for this.
- The Jansche PDF could not be fetched (404), so no empirical gain over Viterbi is reported here.

## 4. Libraries, licences and determinism

### Takeaway
All decoding work can be plain NumPy or Numba and hence deterministic on CPU. Any neural step should be trained once on the GPU, its weights frozen, and inference regenerated on CPU, since PyTorch guarantees nothing across hardware. Avoid `pyhsmm` and `hmmlearn` as dependencies.

### Cited Findings

| Library | Use | Licence | State | Source |
|---|---|---|---|---|
| ruptures | offline change points | BSD-2-Clause | 2.1k stars, 600 commits; last commit not visible | [repo](https://github.com/deepcharles/ruptures) |
| pytorch-crf | linear-chain CRF | MIT | 980 stars, 139 commits, 1 open issue; activity unclear | [repo](https://github.com/kmkurn/pytorch-crf) |
| hmmlearn | HMM, no explicit durations | BSD-3-Clause | "limited-maintenance mode" | [repo](https://github.com/hmmlearn/hmmlearn) |
| pyhsmm | Bayesian explicit-duration HSMM | MIT | not maintained, last tested on Python 3.7 | [ecosyste.ms snapshot](https://awesome.ecosyste.ms/projects/github.com%2Fmattjj%2Fpyhsmm) |
| FACT | action segmentation | MIT per mirror only | YAML configs, pre-trained weights | [repo](https://www.github.com/ZijiaLewisLu/CVPR2024-FACT) |
| PathCRF | possession CRF | not stated in paper | public code | [repo](https://github.com/hyunsungkim-ds/pathcrf.git) |

- PyTorch defines determinism as the same output on the same software and hardware, and does not guarantee reproducibility across releases, commits or platforms — [PyTorch randomness notes](https://pytorch.org/docs/stable/notes/randomness.html).
- `Conv1d` on CUDA is normally non-deterministic and becomes deterministic under `torch.use_deterministic_algorithms(True)`, with `CUBLAS_WORKSPACE_CONFIG` required for cuBLAS-backed ops or a RuntimeError is raised — [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms).
- cuDNN benchmark must be disabled because it may select different algorithms across runs. Deterministic operations are often slower — [PyTorch randomness notes](https://pytorch.org/docs/stable/notes/randomness.html).

### Inferences
- Determinism recipe:
  - Keep frozen weights as a versioned artefact and do not retrain in the regeneration path.
  - Run inference on CPU with fixed thread count and pinned versions of PyTorch, NumPy and BLAS.
  - Round probabilities to a fixed precision, about 6 decimals, before decoding, to absorb tiny floating-point differences.
  - Feed the rounded values to a pure-NumPy decoder and hash the output.
- CPU training is workable for a small TCN at 5 fps on 16 threads, and it is simpler to reproduce, but the same-machine limit still applies.
- `pytorch-crf` is fine for constant transitions, but a feature-dependent transition is not its stock API as far as I could tell. A 40-line custom forward algorithm in PyTorch is likely simpler. Check this.
- HSMM should be custom code: `hmmlearn` has no durations and `pyhsmm` is unmaintained.

### Gaps
- `hsmmlearn` and `pomegranate` could not be confirmed by the search, so nothing on their licence, state or HSMM support is reported.
- Last-commit dates were not visible for any repo.
- No source confirms bit-identical CPU and GPU Conv1d, so assume they differ.

## 5. Sports-specific uses of temporal models

### Takeaway
Two recent works on 7-match Bundesliga-style data show sequence and structure help at this data size, and both report segment-level metrics below their frame metrics. Neither reports possession spell IoU F1, so 0.48-0.50 cannot be benchmarked against the literature.

### Cited Findings
- PathCRF reaches 92.40% team possession frame accuracy and 75.69 on-ball event F1 from player tracking alone, with no ball tracking, on 7 DFL matches. It evaluates on one test match — [PathCRF](https://arxiv.org/html/2602.12080v2).
- Li & Link: 7 TRACAB matches, about 465k frames, 1,257 stable possession sequences, one annotator; frame macro F1 0.76-0.87 vs sequence-level IoT-D F1 0.61-0.79 after post-processing. IoT-D is a custom metric, adopted because of concerns about frame-level boundary precision, and rule-based baselines trailed by 15.2-21.7% frame F1 — [Li & Link 2026](https://arxiv.org/html/2606.09289v1).
- Vidal-Codina et al.: decision-tree possession at frame level, with event detection using possession changes; detection above 90% in most event categories across tournaments and providers. No possession accuracy in the abstract — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v2).
- Transformer point-process models of event sequences can evaluate possession periods, but operate on event data, not tracking frames — [arXiv 2302.09276](https://arxiv.org/pdf/2302.09276).

### Inferences
- PathCRF's possession formulation is the closest external design and worth reading for its masks. Its data appears to be the same dataset as the IDSSE matches in this repo (to confirm), so its splits and code could be compared directly.
- Li & Link's choice to avoid IoU because of boundary imprecision supports the view that sub-second boundaries are the hard part. It also suggests that human-truth boundary noise bounds attainable spellF1.
- Proposed order for the 16 days left, by evidence and cost:
  1. Spell-count sensitivity: tune temperature and costs for spellF1. Hours.
  2. Sampled MBR decoder with a Viterbi grid. 1-2 days, no training.
  3. ASRF-style boundary head, as a better boundary score feeding the decoder, with Gaussian-smoothed boundary targets. 2-3 days.
  4. Low-capacity feature-dependent transitions with masks. 2-3 days.
  5. Small TCN on out-of-fold log-probs. Only if 1-4 plateau, and judged by paired leave-one-match-out.
  Skip DiffAct, ASFormer, FACT, BCN, pure PELT or BOCPD and library HSMMs.
- Validate each step against a re-weighted or 14 s-like duration scenario so the decoder does not lock in on IDSSE's segment density.

### Gaps
- I found no published use of MS-TCN, ASRF or HSMM on team-possession spells, and no published IoU-matched possession metric.
- Whether the 14 s public sample reflects a different annotation convention or a sampling difference is not answerable from these sources, though it decides how much IDSSE-tuned segmentation will transfer.
