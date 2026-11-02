# Deep sequence models and structured decoders for possession spells

Evidence status, 2026-11-01. No source measures spellF1, so every expected gain below is my judgement, not a published or measured number. Papers were read through fetched pages or search summaries. The PDFs of Cohen & Carvalho, Lea et al. and Borghesi et al. could not be read in full (see Gaps). No downloaded code was run. Project facts come from `BITACORA.md`, `RULES.md`, `FAQ.md` and `EVALUATION.md`.

**Main reading.** The project log gives team oracle 0.909 and dead-ball oracle 0.516, and says "the limit is deciding H or A". Decoder-only changes on the same posterior have all failed: MBR −0.005 (1/7), change detector −0.006/+0.013, wider cost grid, and Smart Tagging in the decoder even with perfect labels (−0.001 to +0.005). So the options worth testing bring new H/A evidence or longer context. New decoders alone probably will not reach the gate.

| Rank | Option | Replaces or complements | Expected paired gain (my estimate) | Overfit risk with 7 matches | Determinism | Code and licence |
|---|---|---|---|---|---|---|
| 0 | Oracle decomposition: v3 spells with true labels, and true spells with v3 labels | Diagnostic | None; decides which options below are worth running | None | Existing code | Own |
| 1 | Two-stage HGB with long-range out-of-fold log-odds | Complements | +0.01 to +0.03 | Low to medium; leakage is the main danger | HGB path only | Own; scikit-learn BSD |
| 2 | Spell re-scoring: HGB on spell aggregates, relabel and merge | Complements Viterbi | +0.01 to +0.04 | Medium; about 3,000 correlated spells | HGB path only | Own |
| 3 | Team-grouped set encoder + temporal model, used as an extra feature source | Complements HGB | +0.01 to +0.05, widest spread, highest ceiling | High; domain gap to degraded tracking | Frozen weights, integer inference | Own, ideas from PathCRF (MPL-2.0) and Set Transformer (MIT) |
| 4 | Segmental (semi-Markov) DP with weak duration term | Replaces Viterbi | 0 to +0.02 | Low if the term is weak; duration shift is the risk | Integer DP | Own |
| 5 | Low-capacity context-dependent transition costs | Extends Viterbi | 0 to +0.02 | Medium | Integer DP | Own |
| 6 | MS-TCN-style network on v3 log-probs | Complements or replaces Viterbi | 0 to +0.02 | Medium to high | Frozen weights, integer inference | MS-TCN2 MIT, ASRF MIT |
| 7 | Attention or diffusion segmenters (ASFormer, DiffAct, FACT, LTContext) | Replaces | Not credible at this data size | High | Hard | ASFormer, DiffAct, FACT MIT; LTContext CC BY-NC 4.0 |

Gate statistics: per the log, the gate is +0.03 paired, at least 5/7 matches, no worse on Metrica, with per-match noise about 0.04. If 0.04 is the SD of paired differences, the 7-match mean has SE about 0.015, so +0.03 is about 2 SE before correcting for the number of arms tried. By sign test, 5/7 has one-sided p = 0.227, 6/7 has p = 0.0625 and 7/7 has p = 0.0078 (my arithmetic). Only 7/7 is strong on its own, so limit the number of arms.

## Temporal action segmentation methods: do they suit a small tabular sequence dataset?

### Takeaway
Only a small MS-TCN-style network or a boundary loss is reasonable, and neither is the first thing to try. Published gains come from video benchmarks with 28 or more sequences and rich action scripts. I found no evidence on tabular sequences with 7 sequences, and the project's change-detector result argues against a boundary head as the main bet.

### Cited Findings
- Licences checked today through the GitHub API or the LICENSE file:

| Repo | Licence | Last push | Note |
|---|---|---|---|
| [MS-TCN2](https://api.github.com/repos/sj-li/MS-TCN2) | MIT | 2023-02 | Third-party implementation |
| [ms-tcn](https://api.github.com/repos/yabufarha/ms-tcn) | Not detected by GitHub | 2023-11 | LICENSE file not read |
| [ASRF](https://api.github.com/repos/yiskw713/asrf) | MIT | 2023-04 | Third-party implementation |
| [ASFormer](https://api.github.com/repos/ChinaYi/ASFormer) | MIT | 2022-02 | |
| [DiffAct](https://api.github.com/repos/Finspire13/DiffAct) | MIT | 2023-08 | |
| [FACT](https://api.github.com/repos/ZijiaLewisLu/CVPR2024-FACT) | MIT | 2026-01 | Confirms what the earlier notes only had from a mirror |
| [LTContext](https://raw.githubusercontent.com/LTContext/LTContext/main/LICENSE) | CC BY-NC 4.0 | 2024-06 | Non-commercial; avoid |
| [ActionFormer](https://raw.githubusercontent.com/happyharrycn/actionformer_release/main/LICENSE) | MIT | Not checked | Detection, not segmentation |

- LTContext combines windowed attention for local frames with sparse attention for long-term context — [LTContext search summary](https://openaccess.thecvf.com/content/ICCV2023/html/Bahrami_How_Much_Temporal_Long-Term_Context_is_Needed_for_Action_Segmentation_ICCV_2023_paper.html).
- ActionFormer classifies every moment and regresses boundaries without proposals, built for action localisation — [ActionFormer, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf).
- Adding a class-agnostic boundary channel plus a CDF segment-shape loss gives gains that shrink as the base model gets stronger. Gains computed from Table 2 by the fetch tool: MS-TCN on 50Salads F1@10 +3.48 and F1@50 +5.19, on Breakfast +5.35 and +4.20. FACT on 50Salads +0.28 and +1.04, on Breakfast +1.00 and +0.51. C2F-TCN on 50Salads F1@25 −0.68. No code URL was found — [arXiv 2604.01859](https://arxiv.org/html/2604.01859v1).
- Semi-supervised action-segmentation papers exist for 5-10% labelled videos, and they target unlabelled-video losses. I found no clean supervised TCN curve against training-set size — [Ding & Yao](https://arxiv.org/pdf/2207.08653); [ICC, AAAI 2022](https://arxiv.org/abs/2112.01402).
- Trees stayed state of the art on medium tabular data (about 10K samples) across 45 datasets, explained by irregular targets and uninformative features — [Grinsztajn et al.](https://arxiv.org/abs/2207.08815).
- Earlier notes hold the benchmark table and the HuGaDB result, where MS-TCN and Bi-LSTM had equal accuracy but F1@50 89.9 vs 81.5 — [segmentacion_temporal.md, extract from arXiv 2202.01727](https://arxiv.org/pdf/2202.01727).

### Inferences
- A TCN on v3 log-probs only sees what the HGB already distilled. It can learn long context and spell structure, but it cannot recover information the HGB discarded. Feeding it a few raw signals helps only if they are not already in the 85 features.
- Downgrade the ASRF boundary head proposed earlier. The change detector (−0.006/+0.013) was close to the same idea and failed, and the oracle says labels matter more than boundaries. Run the decomposition first.
- If a TCN is built anyway, use a 2-stage network with 32-64 filters at 5 fps, and take the CDF loss as cheap regularisation. PathCRF's 25 fps run added only 1.52 F1 points at about 12 times the epoch cost, so 5 fps looks sufficient ([PathCRF](https://arxiv.org/html/2602.12080v2)).
- ActionFormer predicts overlapping detections. Spells form a partition of the match, so it does not fit. DiffAct stays excluded because of sampling and data needs.

### Gaps
- No published reduced-data curve for MS-TCN, ASRF, ASFormer, FACT or DiffAct.
- No study applies these models to team-sport tabular features with fewer than 10 sequences.

## Permutation-invariant models over player sets, then a temporal model

### Takeaway
This is the only option with direct evidence of strong ball-free H/A decisions at this data size: PathCRF reached 92.40% frame team accuracy with 5 training matches. It is a high-ceiling, high-risk complement. The set encoder mattered much more than the temporal block, so a simple team-grouped encoder followed by a TCN or GRU is the design to test, as an added feature and not as a replacement for HGB.

### Cited Findings
- PathCRF trained on 5 Sportec matches (379 episodes, 54,683 windows), with 1 validation and 1 test match, and reports frame-level team possession accuracy of 92.40% with no ball input — [PathCRF](https://arxiv.org/html/2602.12080v2).
- Its ablation shows the social module dominating. SAB/PE-SABs: edge accuracy 58.49, event F1 52.34 without CRF and 68.19 with Dynamic Masked CRF. PPE-FPE/Bi-LSTM: 69.15, 66.77 and 75.24. PPE-FPE/PE-SABs: 69.24, 65.34 and 75.69 — [PathCRF](https://arxiv.org/html/2602.12080v2).
- Windows are 10 s, downsampled to 5 fps (50 steps), and the authors say models can be trained from scratch. The PPE module processes each team and the outside nodes separately, while FPE runs one ISAB over all nodes — [PathCRF](https://arxiv.org/html/2602.12080v2).
- PathCRF is MPL-2.0, last pushed 2026-06 — [GitHub API](https://api.github.com/repos/hyunsungkim-ds/pathcrf). Its checkpoints are in `saved/100` to `saved/140`, and the data is the Sportec Open DFL set — [README](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/README.md).
- Ball Radar combines Set Transformers with a hierarchical stage that predicts possession first and then the ball trajectory — [Kim et al., KDD 2023](https://arxiv.org/abs/2306.08206). I could not locate its repository.
- Borghesi et al. encode each trajectory with a TCN, then combine trajectories through a permutation-invariant layer, comparing sum, self-attention and 2D convolutions. Search summary reports 86.2% possession accuracy, above rule-based methods. Full text returned 403 — [Borghesi et al., ESWA 2023](https://iris.polito.it/handle/11583/2976470).
- TacticAI used 7,176 corners and a group-equivariant network built on four reflections of each situation — [DeepMind](https://deepmind.google/blog/tacticai-ai-assistant-for-football-tactics/). The paper is [Nature Communications 2024](https://doi.org/10.1038/s41467-024-45965-x), not read here.
- GNN vs boosted trees on football is mixed. Receiver selection: 0.5387 to 0.5718 accuracy against XGBoost, with a smaller gain on one league — [arXiv 2609.04803](https://arxiv.org/pdf/2609.04803). Pass success: GAT AUC 0.9167 vs XGBoost 0.9124, while boosting beat GNNs on shot blocking and goal tasks — [arXiv 2512.10355](https://arxiv.org/pdf/2512.10355).
- On 7 TRACAB matches, the sequence-level gap of Random Forest against T-GAN on Counter Attack was −41.4% unfiltered and −34.8% after post-processing, and Pure Transformer was −13.1% and −17.8%. Pure GNN was 11.1-22.5% below T-GAN before post-processing — [Li & Link 2026](https://arxiv.org/html/2606.09289v1). The Random Forest there works on frame-level global features, so this does not show that a stacked, temporal HGB would lose.
- Set Transformer reference code is MIT, last pushed 2020 — [GitHub API](https://api.github.com/repos/juho-lee/set_transformer).

### Inferences
- The log blocks PathCRF because it needs identities and goalkeeper, and external code is blocked in this environment. A team-level possession variant needs only two sets of players, which the project already handles through its team-shape features, so a clean-room version avoids both blocks. Confirm that team assignment is reliable enough on the challenge tracking before starting.
- Suggested minimal model, written from scratch:
  1. Per-frame shared MLP on each player's relative position and velocity.
  2. Sum and max pooling per team, then concatenate. Use no attention, because softmax complicates integer inference.
  3. A 2-3 layer dilated TCN or GRU over 20-60 s at 5 fps.
  4. Heads: H/A logit and dead logit.
- Sum pooling is my choice for determinism and capacity. PathCRF's result favours its PPE-FPE design over plain SAB, and I do not know how sum pooling compares. Borghesi's paper may answer this.
- Use the same degradations as v3 (3 degradations × 2 gap patterns), plus x-flip and home-away swap with label swap. Mirroring did nothing for HGB (`symmetric` within ±0.01), but it may matter more for a network, which lacks the HGB's hand-built features.
- The 92.40% is an optimistic bound for this project: one test match, clean tracking, no detection noise, and possibly a different label convention. Compare it with v3's own frame H/A accuracy on IDSSE before investing, because that number was not in the documents I read.
- PathCRF's pre-trained checkpoints would leak if stacked in leave-one-match-out CV, since they were trained on 5 of what appear to be the same 7 matches. Rules allow pre-trained models only if organisers can access them, and the log's block on external code remains.
- Keep the network's output as a complement. Average its log-odds with HGB using one global weight, then pass it to v3's decoder. This needs only 7 outer fits. Full stacking into HGB needs 7 × 6 = 42 fits; if one fit took 30 minutes, that is 21 hours of GPU, which is an assumed fit time, not a measurement.

### Gaps
- I found no study showing a set or graph model beating GBDT on possession with fewer than 10 matches, and no head-to-head on noisy tracking.
- Whether PathCRF's accuracy depends on clean team labels is not reported, because the paper has no ablation without them.
- Borghesi's method comparison and data size could not be read.
- TacticAI's symmetry gain on small data is known only from secondary sources, and it concerns corners rather than continuous play.

## Semi-Markov and segmental models, neural CRF, learned transition costs, spell re-scoring

### Takeaway
Segment re-scoring is the most promising item here because it pools H/A evidence across a whole spell, which attacks the oracle gap directly. Duration priors and learned transitions are cheap but have low prior gain: PathCRF's feature-dependent transitions added about 1.9 points, and free dense transitions lost points.

### Cited Findings
- PathCRF event F1: non-CRF 65.34, Viterbi 73.14, static dense 68.78, dynamic dense 67.82, static masked 73.83, dynamic masked 75.69. Its transition score is an MLP over concatenated edge embeddings, trained by negative log-likelihood with the forward algorithm — [PathCRF](https://arxiv.org/html/2602.12080v2).
- Semi-CRFs score whole segments and so can use segment features such as length, and the best segmentation is found with a Viterbi-style recursion — [Exploring Segment Representations](https://arxiv.org/pdf/1604.05499). Segmental RNNs equal a semi-CRF with neural local features — [SRNN](https://arxiv.org/pdf/1511.06018).
- Semi-CRF inference is quadratic in sequence length — [Filtered Semi-Markov CRF](https://arxiv.org/pdf/2311.18028).
- Lea et al. jointly segment and classify with a semi-Markov model, using per-segment features and a constrained inference much faster than standard semi-Markov inference. Numbers not read — [Segmental Spatiotemporal CNNs](https://arxiv.org/pdf/1602.02995).
- HSMMs remove the implicit geometric duration of HMMs, and HMMs are the special case with length-one segments. This comes from a search summary that did not tie each statement to one paper, so check before quoting — [Dai et al., likely source](https://hanjun-dai.github.io/pdf/DaiDaiZhaetal17.pdf); [arXiv 1909.05800, also in those results](https://arxiv.org/pdf/1909.05800).
- ASRF cuts at boundaries and gives each segment the majority class of the frame branch — [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866). This is the simplest form of cut-then-classify.
- In Li & Link, post-processing raised T-GAN sequence-level F1 for intentions from 0.68 to 0.79, and for Finishing from 0.45 to 0.68 — [Li & Link 2026](https://arxiv.org/html/2606.09289v1).
- pytorch-crf is MIT, last pushed 2024-06 — [GitHub API](https://api.github.com/repos/kmkurn/pytorch-crf).

### Inferences
- **Spell re-scoring design.**
  1. Candidates: v3 spells, plus a lower-cost Viterbi run (for example switch cost 2) to allow over-segmentation.
  2. Features per candidate: mean, median and max of log-odds, ball-seen fraction, duration, start and end ball state, spell-level shape and control aggregates, and the neighbouring spells' labels.
  3. Target: majority true label of the candidate, from decoded out-of-fold spells. Do not train on true spells, because their boundaries are too clean.
  4. Relabel, merge equal neighbours, and optionally solve a one-dimensional DP to keep or merge cuts.
  5. Use a small HGB with strong leaf regularisation.
- Sample size is about 437 × 7 ≈ 3,000 true spells, and more with degraded copies, though these are not independent. Project log counts differ by label set (`ctrl` has 227 per match).
- Spell re-scoring cannot fix a wrong boundary unless the candidates contain the right cut. The decomposition (rank 0) gives its ceiling: if v3 spells with true labels score far above 0.556, this option has room.
- **Segmental DP.** At 5 fps with a 60 s cap, L = 300 and T is about 40,000, so states × T × L is about 36 million segment evaluations, tractable in Numba with prefix sums. This is my arithmetic.
- Duration term: use a weak parametric hazard, because spell lengths differ widely: IDSSE 7.2 s, Metrica about 9 s, v3 on the challenge 11.2 s, public 5 minutes 14.2 s. The term should mainly penalise implausibly short spells, and must pass a 2× duration-scale test.
- **Context transitions.** Use 5-10 features at most (ball-state change, ball-speed jump, inter-team distance change), trained as one logistic model on out-of-fold spell boundaries, and subtract its score from the switch cost. PathCRF's +1.86 gain was on a much bigger edge state space, and the dense-vs-masked loss warns that capacity hurts. The change detector's failure is a further reason to expect little.
- Neural CRF training as a final layer belongs only inside the network of rank 3. Its benefit there would be the same as Viterbi's over independent frames.

### Gaps
- No head-to-head of HSMM against HMM on segmental F1 in action segmentation or sport.
- No published spell re-scoring on team possession.
- Lea et al. numbers could not be read because the PDF did not parse.

## Stacking: sequence outputs into HGB, or two-stage HGB with lagged out-of-fold probabilities

### Takeaway
Two-stage HGB is the best first experiment: it adds true long-range context, needs no new dependency, keeps the deterministic path and has the oldest supporting literature. The price is nested cross-validation, because any leakage would look like a large gain.

### Cited Findings
- Stacked sequential learning makes a base learner aware of nearby labels. It targets sequential partitioning, meaning long runs of identical labels. It improved non-sequential learners, could help CRFs, and a stacked maximum-entropy learner generally beat CRFs on those tasks — [Cohen & Carvalho, IJCAI 2005](https://mlanthology.org/ijcai/2005/cohen2005ijcai-stacked).
- A multi-scale extension of stacked sequential learning exists — [CVC](https://www.cvc.uab.es/blog/2011/11/04/multi-scale-stacked-sequential-learning/); [Generalized Stacked Sequential Learning](https://tesisenred.net/handle/10803/285969).
- Earlier project arms `two_heads`, `self_train`, `seeds3` and `lightgbm` stayed within ±0.01 — project log, `BITACORA.md`. None of them is two-stage context stacking, going by their names; confirm in `possession/experiments/improve.py`.

### Inferences
- Design:
  1. Level 0: v3 HGB, with out-of-fold log-odds per frame.
  2. Level 1 features at multiple scales (±1, 5, 15, 60 s, forward and backward): mean and max log-odds, entropy, sign flips, distance to last confident H or A frame, gap-adjusted confidence.
  3. Add Viterbi-derived features from level-0 outputs: current spell's team, age and remaining length. These are new compared with a limited frame window.
  4. Level 1 HGB with v3's features or a reduced subset, then the same Viterbi.
- Nesting: for each of 7 outer test matches, train level 0 on 6 matches and produce level-1 training inputs by inner leave-one-match-out on those 6. That is 7 × 7 = 49 HGB fits including outer ones, so measure one arm on a reduced setup first. For the final model, level-1 is trained on inner out-of-fold outputs from all 7, and level-0 is refit on all 7.
- Level-1 inputs for the test match come from a model trained on more data than the inner fold models, so its confidence shifts slightly. Calibrating log-odds with one temperature reduces this.
- Stop after two levels. Each level uses the same 7 matches, and the extra-stage signal is correlated with the first.
- `sample_weight` of ones changes the trees (project log), so keep the training path exact when comparing.
- Sequence-model outputs into HGB need the same nesting and are much more expensive than HGB-only stacking, so first test the single-weight blend given above.

### Gaps
- The Cohen & Carvalho PDF did not parse, so the exact inner cross-validation procedure and result numbers are not quoted. My understanding that stage-1 training predictions must be out-of-fold comes from general practice, and not from that text.
- No source reports stacking gain on sports tracking.

## Ways to optimise a segmental IoU metric directly

### Takeaway
I found no differentiable or exactly solvable form of one-to-one IoU-matched F1 averaged over thresholds. The structured-learning literature handles count-based F1, and it can only tune a few parameters. Given the MBR result, treat direct optimisation as a final polish worth 0 to +0.01, not as a route to +0.03.

### Cited Findings
- Segmental F1 was introduced with temporal convolutional action segmentation: it penalises over-segmentation and tolerates small shifts — [Lea et al. 2016](https://arxiv.org/pdf/1611.05267).
- Structural SVM training for sequential labelling under F1 and other contingency-table losses reports gains over HMMs and decomposable-loss SVMs on TUM Kitchen and CMU activity data — [UTS](https://opus.lib.uts.edu.au/handle/10453/35135); [TNNLS version](https://opus.lib.uts.edu.au/handle/10453/132470).
- Exact loss-augmented inference exists for measures such as F-beta and IoU on sequence segmentation when they are expressed through a high-order potential with decomposable internal structure — [Efficient Exact Inference](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c351t-9db238add92346471c1bae67b71db5a284d7916bb07a287c0c8482fe5b7174e73). Ranjbar et al. handle measures based on false-positive and false-negative counts — [Ranjbar et al.](https://www2.cs.sfu.ca/~mori/research/papers/ranjbar_pami12.pdf).
- Auto Seg-Loss searches parameterised surrogates of segmentation metrics by replacing non-differentiable operations, for image segmentation — [Auto Seg-Loss](https://ar5iv.arxiv.org/html/2010.07930).
- The CDF loss regularises mass within each true segment, using a uniform target over a segment excluding a margin near boundaries — [arXiv 2604.01859](https://arxiv.org/html/2604.01859v1).
- The metric matches predicted and true possessions one-to-one by temporal IoU and averages F1 over thresholds 0.3 to 0.7 — project `EVALUATION.md`.

### Inferences
- Those losses count frames or labels, so they are not spellF1. spellF1 depends on a global matching, and no source gives loss-augmented inference for it.
- Practical routes, by low effort:
  1. Spell-balanced sample weights, with each true spell given equal total weight, which approximates a per-segment recall. Check whether the rejected `weighted` arm already did this.
  2. For the few parameters in a re-scorer or DP, use a derivative-free search on nested spellF1, with at most 2-3 free parameters. More would repeat the rejected cost grid and invite selection bias.
  3. In a network, add a soft IoU per true spell, computed as soft intersection and union on the matching class, to approximate a segment-level recall. It is my own construction and untested.
- The MBR failure is informative: it already used spellF1 on posterior samples. If samples were well calibrated, MBR should have helped, so its failure points to weak or poorly calibrated posteriors. Better inputs are a stronger lever than a better objective.

### Gaps
- No published differentiable surrogate for IoU-matched segmental F1 was found. Full-text search of the 2604.01859 citing papers was not done.
- Whether exact loss-augmented inference applies to a matching-based IoU is unknown.

## Making a neural model byte-identical on CPU

### Takeaway
No framework documents a cross-machine bitwise guarantee. The safest route is to freeze weights trained once on GPU, and run inference in the project's own integer, loop-based Numba or NumPy code with no BLAS or framework in the regeneration path. Local tests confirm BLAS output changes with thread count in the pinned environment, while explicit loops stayed identical. ONNX Runtime and PyTorch inference could not be tested here.

### Cited Findings
- "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms", and results may not match between CPU and GPU even with identical seeds — [PyTorch notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html).
- oneDNN's deterministic attribute guarantees bitwise identical results for multiple executions of the same primitive on a given platform, and the default permits non-identical outputs. Bit-wise equality is expected only with identical hardware and software environment — [oneDNN](https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-2/primitive-attributes-deterministic.html).
- ISA can be capped with `ONEDNN_MAX_CPU_ISA` only if the build enables it — [oneDNN dispatcher](https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-1/cpu-dispatcher-control.html).
- MKL_CBWR selects a code branch — [Intel developer guide](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-2/set-env-var-for-cond-numerical-reproducibility.html). Intel moderators say it needs the same thread count, with `STRICT` covering only a limited set of functions. A forum user still got different results across machines — [Intel forum](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/MKL-results-reproducibility/m-p/1286794).
- ONNX Runtime's threading guide covers `intra_op_num_threads`, default physical cores, sequential as default execution, and spinning, and says nothing on determinism — [ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
- `use_deterministic_compute` is per-kernel and mainly concerns GPU kernels. Commit notes describe warnings for non-deterministic kernels, and a CERN group reported non-determinism in GPU inference despite it — [ORT commit](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/15cb2f5a8aa1bdbbb71de9560dc8835a943264d5); [CERN note](https://indico.cern.ch/event/1572328/contributions/6624197/note/).
- A single anecdotal issue reports run-to-run TopK differences on ORT 1.19.2 CPU on Windows with no `SessionOptions` — [ORT #28018](https://github.com/microsoft/onnxruntime/issues/28018).
- Integer quantisation is not automatically CPU-independent. On x86-64 with AVX2 or AVX512 without VNNI, ORT uses VPMADDUBSW for U8S8, which can saturate, while VNNI and ARM do not have this issue. `reduce_range` or U8U8 are the mitigations — [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
- One paper found identical training deterministic under the same CPU thread count, and non-deterministic under a different one — [ISSRE 2021](https://yuleisui.github.io/publications/issre21.pdf).

**Local measurements, this PC, synthetic data, nothing from the challenge, no files created:**
- Pinned `.venv` (Python 3.12, numpy 2.5.3): `x @ w @ A` on a 2000×64 matrix gave SHA-256 prefix `31deea03ba` with 1 and 4 threads, `cf1a21e28d` with 8 and `23df802a50` with 16. So BLAS matrix products are thread-count dependent here.
- System Python 3.13 (numpy 2.4.6, numba 0.67): a 4-layer dilated convolution, 32,000 × 64, written as explicit Numba `prange` loops over frames, gave the same hash at 1 and 8 threads in float64 (`3fe69ef9fb50`) and in int64 fixed-point (`4ab324c745db`). The BLAS product in that same run differed between 1 and 8 threads.
- A float64 Numba convolution, and NumPy `exp`, `log`, `tanh`, `cumsum`, `sum` and `dot`, gave equal hashes across default settings, `NUMBA_CPU_NAME=generic`, and `NPY_DISABLE_CPU_FEATURES` for AVX2, FMA3 and AVX512. This is weak evidence: I did not confirm that those settings changed the code path taken on this CPU.
- `torch` and `onnxruntime` are absent from `.venv`, and `torch` fails to import in system Python, so neither was tested. Nothing was installed.

### Inferences
- **Recommended path, in order.**
  1. Train once on GPU. Treat weights as a versioned artefact trained only on external matches. No challenge data enters training, which keeps them publishable.
  2. Convert to fixed point: `w_int = round(w × 2^12)`. Evaluate the quantised model through CV, since this is the delivered model.
  3. Use only ops with exact integer forms: conv, linear, ReLU, sum, max, right shift. Avoid softmax in the network, and output logits. For GRUs and attention, use a lookup table.
  4. Implement inference in Numba with `int64` and per-frame `prange`, with no shared reductions. Check overflow against worst-case bounds.
  5. Decode with integer DP costs, with ties broken by a fixed rule.
  6. Pin `numba` and `llvmlite` in `requirements-solution.txt`, set `NUMBA_CPU_NAME=generic`, and turn off Numba's cache.
  7. Add a SHA-256 check to the pipeline for logits and CSV. Run it at 1, 4, 8 and 16 threads, and on the clean machine that is already pending.
- Inputs still float: features come from float pipelines, so they should be quantised in one fixed function before the network. Any float difference earlier in the pipeline would already threaten the current v3 CSV, so this adds no new class of risk.
- **Correction to earlier notes.** Rounding probabilities to 6 decimals reduces differences but gives no guarantee, because any value near a rounding edge can still flip. Use exact integer arithmetic in the network instead, or keep a cross-machine check.
- If a framework is unavoidable, use ONNX Runtime with explicit `SessionOptions`, `intra_op_num_threads=1`, `inter_op_num_threads=1`, sequential execution, a pinned version and a pinned wheel. Avoid U8S8 quantisation, or use `reduce_range`. Treat this as same-machine reproducibility until tested elsewhere. My local tests cover only explicit loops and BLAS, not ORT.
- Any retraining path on GPU cannot be part of "run the repository", because PyTorch offers no cross-platform guarantee and CUDA adds another source. The rule says "the same file", so inference from committed weights is the viable design. CPU-only single-thread training would be repeatable on one machine at most.
- Existing NumPy code that uses `@`, `np.dot` or `np.linalg` in the regeneration path deserves a check, or `OPENBLAS_NUM_THREADS` must be fixed in the pipeline. I did not audit `possession/`.

### Gaps
- No official ONNX Runtime statement on bitwise CPU determinism was found, whether across thread counts or machines.
- ORT, PyTorch CPU and the U8U8 path were not tested, so all guidance on them comes from documentation or anecdotes.
- All local tests ran on one CPU, so cross-CPU behaviour is not verified. The float loop test was not repeated in the pinned `.venv`.
- Whether committed weights are acceptable under rule 3 is my reading of "running your repository"; `RULES.md` does not address it, and the project chose not to consult organisers on reproducibility.
