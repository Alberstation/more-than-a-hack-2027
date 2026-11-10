# Ball possession from automatic tracking — technical document

**Team Barca Reborn** · More than a Hack 2027 · repository
[github.com/Alberstation/more-than-a-hack-2027](https://github.com/Alberstation/more-than-a-hack-2027)

Barça–Athletic, 199,319 tactical-video frames, one label per frame: `H`, `A`, `D` (dead ball) or
`X` (not graded). Submission: version **v5**. Code: [`possession/`](../possession), run from
[SOLUTION.md](../SOLUTION.md).

## 1. Approach

**The problem as we saw it.** The challenge tracking is automatic and incomplete: the ball is missing
from 41 % of the graded frames, players come and go as tracklets, and there are no identities. No
label for the match may be used. So the model has to learn possession on *other* matches, and those
matches have to be made to look like this tracker before learning from them.

**Pipeline (v5).**

1. **Training data: degraded public matches.** Seven Bundesliga matches from the IDSSE/DFL open
   dataset have clean tracking with possession and ball status per frame. We degrade them to match
   the challenge tracker. The ball is hidden in gaps drawn from the length distribution of the
   challenge's own gaps, under two patterns: uniformly at random, or concentrated in dead-ball
   stretches as the challenge's seem to be. We also drop player slots for 1–30 s stretches (7 % of
   slot time), add slowly drifting position error (0.25 m), and make the dead ball *move*
   (IDSSE freezes it, the challenge tracker does not). Each match is degraded three times under both
   patterns.
2. **Labels for controlled possession** (`control.py`, §4): the DFL field, corrected so that a team
   that never plays the ball does not get possession.
3. **Two per-frame models, mixed.**
   - A **gradient-boosted tree model** (scikit-learn `HistGradientBoostingClassifier`) on 85
     identity-free features: distances of each team to the ball, who is nearest and when that
     changes, ball kicks, team shape (quantiles, spread, advance), pass chains, retention. Weight 0.25.
   - A **set encoder**: per team, a shared MLP over players (position, velocity, offset to the ball)
     pooled by mean, max and closeness to the ball, followed by a dilated temporal convolution seeing
     about ±50 s at 5 steps/s; 451 k parameters, 3 seeds averaged. Weight 0.75. Trained in PyTorch
     on GPU; at inference it runs in NumPy float64, so the submission needs no GPU and no torch.
4. **Decoding.** One three-state Viterbi pass per half over the mixed log-probabilities, with a cost
   of 20 for each change of possession and 20 for each change to or from dead ball. This turns
   flickering per-frame decisions into possessions, which is what `spellF1` counts.

**Why this rather than something else.** Rules on the nearest player fragment possession wherever
the ball is missing (the starter baseline). A model trained on clean tracking learns signals that
the challenge does not have: a frozen dead ball, a ball that is always visible. The gains came from
closing that gap: matching the gap pattern (+0.05 to +0.08 spellF1 on held-out matches), a moving
dead ball (+0.03 to +0.08), labels that follow the definition of controlled possession, and the set
encoder's long temporal context (+0.04 to +0.07). Every constant (decoder costs, mixture weight) is
chosen by leave-one-match-out validation on IDSSE, never on the challenge.

**Validation results**, `spellF1` against controlled-possession labels. The two figures are the two
gap patterns (uniform / in dead ball). IDSSE is leave-one-match-out over seven matches; Metrica is a
different provider, never used for fitting or choosing.

| Version | What it adds | IDSSE | Metrica |
|---|---|---|---|
| v4 | HGB, 3 degradations × 2 gap patterns, moving dead ball, Viterbi | 0.690 / 0.721 | 0.669 / 0.697 |
| **v5** | HGB on control labels + set encoder (weight 0.75), Viterbi (20, 20) | **0.767 / 0.779** | **0.672 / 0.693** |

The public five minutes were used only to check the file and to understand the labels: v5 scores
0.707 on them (v4 0.637). They are 17 possessions and, being the stretch with the most ball, they
do not represent the match; nothing was chosen on them.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD tracking | **Used**: the only input to the models. |
| Smart Tagging | **Used** only for the two `Kick Off (Start)` frames: before them each half is `D`. Using its set pieces or phases inside the decoder was simulated and dropped (−0.001 to +0.005, even with perfect labels). |
| Ball gaps of the challenge tracking | **Used** as input statistics: their lengths, and how often the ball is missing when the v2 model calls play dead (a prediction, no label), drive the degradation. The ball's speed and velocity-change quantiles were measured to check the degradation (§5). |
| Homography, broadcast tracking and video | Not used. The ATD positions are already in pitch metres; the broadcast feed is not synchronised and no video file was available on our machine. A ball recovered from video (WASB) was simulated by restoring part of the gaps: +0.026, under our acceptance threshold. |
| Public five-minute ground truth | Format check and error analysis only. |

**Methods tried and dropped**, all measured on held-out IDSSE matches and Metrica, accepted only with
+0.03 paired, at least 5 of 7 matches up, and no loss on Metrica:

- change-point detector feeding the decoder;
- minimum Bayes risk decoding;
- re-scoring segments by mean probability, and a two-stage HGB on windowed probabilities;
- geometry features: Voronoi, pressure, projected arrival;
- per-possession loss weights and learned transitions in the encoder, to recover short possessions;
- explicit touch evidence on a ball blurred like the challenge's: +0.011 for the HGB alone,
  nothing once mixed with the encoder;
- Metrica as a second training provider, each match held out in turn: +0.01 to +0.02 when added
  to the encoder (an unseen IDSSE match and an unseen Metrica game both improve), below our
  threshold, so v5 keeps Metrica for validation only;
- a minimum length for dead-ball stretches inside the Viterbi decoder: no effect, because the
  decoder costs already rule out stoppages shorter than 2 s;
- LightGBM, more seeds, self-training on the challenge.

Learned CRF models from the literature (PathCRF) need player identities and goalkeepers, which the
challenge does not provide. **AI use:** the code was written with an AI coding assistant (Claude
Code). No language model is part of the pipeline.

## 3. Training data

| Dataset | Source and licence | Who labelled it, and how | Amount | Use |
|---|---|---|---|---|
| IDSSE / DFL open data | Bassek et al., *Scientific Data* 2025, CC-BY 4.0; downloaded by `possession/idsse.py` | DFL/Sportec: TRACAB optical tracking with `BallPossession` and `BallStatus` per frame, and official event data from trained live operators | 7 Bundesliga matches, ~1 M frames at 25 fps | Fitting the HGB and the encoder; validation (leave one match out) |
| Metrica Sports sample data | `github.com/metrica-sports/sample-data`, games 1–3 | Metrica: tracking and event data provided by the company; possession built from its events | 3 matches | Validation; tested as extra training data (§2), not used by v5 |

Nothing comes from the provided game: its tracking is used as input only, at inference and for the
degradation statistics above. SkillCorner open data was downloaded and not used: the learning curve
had flattened after 4–6 IDSSE matches.

## 4. Possession rules implemented

The challenge defines possession as **controlled**. A contested or loose ball stays with the last
team, and a touch, deflection or aerial duel without control does not change possession. The DFL
`BallPossession` field changes on every touch (30 % of its possessions last less than 2 s), so we
relabel it (`possession/control.py`):

- **Gain of control.** A spell of a team that plays the ball neither in a pass/play event nor in a
  shot within 1.5 s of it is a touch, a deflection or a won duel without control. It is handed back
  to the team before it. The 1.5 s window was fixed before any score was seen and never tuned.
- **Loose and contested balls** stay with the last controlling team. This follows from the rule
  above, and the decoder cost enforces it: changing possession needs sustained evidence.
- **Dead ball (`D`)**: from DFL's `BallStatus`, unchanged. A restart (throw-in, corner, free kick,
  goal kick, kick-off) ends `D` when the ball is played. Each half is `D` until its kick-off.
- **Not graded (`X`)**: everything outside the two halves of the video (before the match, halftime,
  after the final whistle).
- **Decoder.** A possession must outweigh a cost of 20 to appear. A 1 s possession needs a mean
  probability of about 0.83, so very brief spells are merged into their neighbours.

An audit of these labels on clean tracking found that the short controlled possessions start with a
ball touch by their own team as often as long ones (87–90 %), and survive shifts of the event clock
of ±0.5 s (95–97 %). They are real possessions, not alignment artefacts.

## 5. Limitations

- **Short possessions are missed.** Recall of possessions under 2 s is about 20 % (2–4 s: about
  50 %; over 16 s: 91 %). In those spells the model gives the true team about 0.45: the opponent is
  just as close, and tracking alone does not say who touched the ball. Neither reweighting the loss,
  learned transitions, nor explicit touch features moved this. As a result v5 reports about 3.2
  possessions per live minute, against 4.2 in the public window.
- **Training labels are an approximation.** The challenge ground truth comes from corrected
  tracking by its own rules; ours come from DFL events through the rule in §4. Where the two
  definitions differ, we inherit the difference.
- **Simulated tracker.** The degradation reproduces the gap pattern (including where gaps start:
  near the lines, not near players), the moving dead ball and the ball's kinematics (velocity-change quantiles within 10 % after a 0.24 s blur, which v5 tolerates:
  0.764 / 0.784 on the blurred ball). Our dead ball still moves more calmly than the challenge's
  (median 1.45 vs 2.7 m/s), and the challenge loses the ball mostly when it is slow (3.3 vs 6.0 m/s),
  which our gaps do not yet reproduce.
- **One provider for training.** Across providers we lose about 0.1 (IDSSE → Metrica). The
  literature reports similar or larger drops.
- **With more time:** labelled matches from an automatic, broadcast-style tracker (the closest
  match to this domain), the ball recovered from the video to fill gaps, and per-player touch
  attribution from video for the sub-2-second possessions.

## 6. Reproduction

Python 3.12, CPU, pinned dependencies (`requirements-solution.txt`). The match data must be in
`data/` as distributed; nothing from it is in the repository.

```bash
pip install -r requirements-solution.txt
python possession/idsse.py                  # downloads the 7 IDSSE matches and their events
python possession/v5.py                     # fits the HGB on degraded IDSSE (deterministic)
python possession/pipeline.py --out "Barca Reborn.csv"
python evaluation/validate_submission.py "Barca Reborn.csv"
```

The set-encoder weights are the only trained artefact in git (`possession/models/`, 3 × 1.8 MB,
fitted on IDSSE only). GPU retraining is not bit-for-bit repeatable, which is why we commit the
weights. They can be retrained with `possession/experiments/encoder_train.py` in a separate torch
environment. Inference is single-threaded float64, and the CSV is identical from run to run:
SHA-256 `866cb21aea0c890598519e9f819240362eb3a48ef4be9038fc7faf2bc8a32dcc`.
All experiments and their measurements are logged in `BITACORA.md`.
