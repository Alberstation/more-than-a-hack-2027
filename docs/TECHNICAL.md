# Ball possession from automatic tracking — technical document

**Team Barca Reborn** · More than a Hack 2027 · repository
[github.com/Alberstation/more-than-a-hack-2027](https://github.com/Alberstation/more-than-a-hack-2027)

Barça–Athletic, 199,319 tactical-video frames, one label per frame: `H`, `A`, `D` (dead ball) or
`X` (not graded). Submission: version **v6**. Code: [`possession/`](../possession), run from
[SOLUTION.md](../SOLUTION.md).

## 1. Approach

**The problem as we saw it.** The challenge tracking is automatic and incomplete: the ball is missing
from 41 % of the graded frames, players come and go as tracklets, and there are no identities. No
label for the match may be used. So the model has to learn possession on *other* matches, and those
matches have to be made to look like this tracker before learning from them.

**Pipeline (v6).**

1. **Training data: degraded public matches from two providers.** Seven Bundesliga matches from
   the IDSSE/DFL open dataset and the three Metrica Sports sample games. We degrade them to match
   the challenge tracker. The ball is hidden in gaps drawn from the length distribution of the
   challenge's own gaps, under two patterns: uniformly at random, or concentrated in dead-ball
   stretches as the challenge's seem to be. We also drop player slots for 1–30 s stretches (7 % of
   slot time), add slowly drifting position error (0.25 m), and make the dead ball *move* (the
   providers freeze or drop it, the challenge tracker does not). Each match is degraded three times
   under both patterns.
2. **Labels for controlled possession** (`control.py`, §4): the providers' possession, corrected
   so that a team that never plays the ball does not get possession.
3. **Two per-frame models, mixed.**
   - A **gradient-boosted tree model** (scikit-learn `HistGradientBoostingClassifier`) on 85
     identity-free features: distances of each team to the ball, who is nearest and when that
     changes, ball kicks, team shape (quantiles, spread, advance), pass chains, retention. Fitted on
     IDSSE. Weight 0.25.
   - A **set encoder**: per team, a shared MLP over players (position, velocity, offset to the ball)
     pooled by mean, max and closeness to the ball, followed by a dilated temporal convolution seeing
     about ±50 s at 5 steps/s; 451 k parameters, 3 seeds averaged. Fitted on IDSSE and Metrica.
     Weight 0.75. Trained in PyTorch on GPU; at inference it runs in NumPy float64, so the
     submission needs no GPU and no torch.
4. **Decoding.** One three-state Viterbi pass per half over the mixed log-probabilities, with a cost
   of 14 for each change of possession and 20 for each change to or from dead ball. This turns
   flickering per-frame decisions into possessions, which is what `spellF1` counts.

**Why this rather than something else.** Rules on the nearest player fragment possession wherever
the ball is missing (the starter baseline). A model trained on clean tracking learns signals that
the challenge does not have: a frozen dead ball, a ball that is always visible. The gains came from
closing that gap: matching the gap pattern (+0.05 to +0.08 spellF1 on held-out matches), a moving
dead ball (+0.03 to +0.08), labels that follow the definition of controlled possession, the set
encoder's long temporal context (+0.04 to +0.07), and a second provider in the encoder's training
(+0.01 to +0.02). Every constant (decoder costs, mixture weight) is chosen by leave-one-match-out
validation on external data, never on the challenge.

**Validation results**, `spellF1` against controlled-possession labels. The two figures are the two
gap patterns (uniform / in dead ball). IDSSE: each of the seven matches held out in turn. Metrica:
v5 never saw it; for v6, each game is held out from the encoder's training in turn.

| Version | What it adds | IDSSE | Metrica (games 1–3) |
|---|---|---|---|
| v4 | HGB, 3 degradations × 2 gap patterns, moving dead ball, Viterbi | 0.690 / 0.721 | — |
| v5 | HGB on control labels + set encoder (weight 0.75), Viterbi (20, 20) | 0.767 / 0.779 | 0.680 / 0.709 |
| **v6** | v5 with Metrica in the encoder's training, Viterbi (14, 20) | **0.780 / 0.798** | **0.691 / 0.714** |

v6 beats v5 on an IDSSE match it never saw in 4 of 7 matches with uniform gaps and in all 7 with
gaps in dead play, and on each held-out Metrica game. The public five minutes were used only as a
check of the file; nothing was fitted or chosen on them.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD tracking | **Used**: the only input to the models. |
| Smart Tagging | **Used** only for the two `Kick Off (Start)` frames: before them each half is `D`. Using its set-piece windows inside the decoder was tested on external matches and dropped (no gain). |
| Ball gaps of the challenge tracking | **Used** as unlabelled input statistics: the length distribution and share of the tracker's missing-ball gaps set the degradation applied to the external training matches. |
| Homography, broadcast tracking and video | Not used by the models. The ATD positions are already in pitch metres; the broadcast feed is not synchronised. The tactical video was tested as an input only, with nothing trained or tuned on it. The pretrained WASB ball detector (MIT, its soccer weights and default threshold), run on whole frames and on 2×2 crops, put its detection within 2 m of the ATD ball on at most 21.6 % of the frames where the ATD sees it, so filling the gaps with it would add more noise than ball. A fixed-rule whistle detector on the audio found 3 of 6 kick-offs against a 28 % chance rate. Neither is used. |
| Public five-minute ground truth | File check only. |

**Methods tried and dropped**, all measured on held-out external matches, accepted only with +0.03
paired, at least 5 of 7 matches up, and no loss on Metrica (v6 was accepted below that threshold
because it improved on every criterion, including an unseen match of each provider):

- change-point detector feeding the decoder;
- minimum Bayes risk decoding;
- re-scoring segments by mean probability, and a two-stage HGB on windowed probabilities;
- geometry features: Voronoi, pressure, projected arrival;
- per-possession loss weights and learned transitions in the encoder, to recover short possessions;
- explicit touch evidence on a ball blurred like the challenge's: +0.011 for the HGB alone,
  nothing once mixed with the encoder;
- Metrica in the HGB's training: no consistent gain on unseen IDSSE matches (−0.005 / +0.009);
- a minimum length for dead-ball stretches inside the Viterbi decoder: no effect, because the
  decoder costs already rule out stoppages shorter than 2 s;
- ball gaps that start where the challenge loses the ball (slow balls, near the lines): not needed,
  the models score the same or better under them (Metrica 0.715 against 0.709);
- LightGBM, more seeds.

Learned CRF models from the literature (PathCRF) need player identities and goalkeepers, which the
challenge does not provide.

## 3. Training data

| Dataset | Source and licence | Who labelled it, and how | Amount | Use |
|---|---|---|---|---|
| IDSSE / DFL open data | Bassek et al., *Scientific Data* 2025, CC-BY 4.0; downloaded by `possession/idsse.py` | DFL/Sportec: TRACAB optical tracking with `BallPossession` and `BallStatus` per frame, and official event data from trained live operators | 7 Bundesliga matches, ~1 M frames at 25 fps | Fitting the HGB and the encoder; validation (leave one match out) |
| Metrica Sports sample data | `github.com/metrica-sports/sample-data`, games 1–3; no formal licence, its README offers the data to the analytics community and asks for acknowledgement, given here | Metrica: tracking and event data provided by the company; possession built from its events with the rule in §4 | 3 matches, ~430 k frames | Fitting the encoder (v6); validation (leave one game out) |

No label of the provided game is used: its tracking is an input only, at inference and for the
degradation statistics above. SkillCorner open data was downloaded and not used: the learning curve
had flattened after 4–6 IDSSE matches.

## 4. Possession rules implemented

The challenge defines possession as **controlled**. A contested or loose ball stays with the last
team, and a touch, deflection or aerial duel without control does not change possession. The DFL
`BallPossession` field changes on every touch (30 % of its possessions last less than 2 s), so we
relabel it (`possession/control.py`; Metrica's event-built possession goes through the same rule):

- **Gain of control.** A spell of a team that plays the ball neither in a pass/play event nor in a
  shot within 1.5 s of it is a touch, a deflection or a won duel without control. It is handed back
  to the team before it. The 1.5 s window was fixed before any score was seen and never tuned.
- **Loose and contested balls** stay with the last controlling team. This follows from the rule
  above, and the decoder cost enforces it: changing possession needs sustained evidence.
- **Dead ball (`D`)**: from the provider's ball status, unchanged. A restart (throw-in, corner,
  free kick, goal kick, kick-off) ends `D` when the ball is played. Each half is `D` until its
  kick-off.
- **Not graded (`X`)**: everything outside the two halves of the video (before the match, halftime,
  after the final whistle).
- **Decoder.** A possession must outweigh a cost of 14 to appear, so very brief spells with weak
  evidence are merged into their neighbours.

An audit of these labels on clean tracking found that the short controlled possessions start with a
ball touch by their own team as often as long ones (87–90 %), and survive shifts of the event clock
of ±0.5 s (95–97 %). They are real possessions, not alignment artefacts.

## 5. Limitations

- **Short possessions are missed.** Recall of possessions under 2 s is about 20 % (2–4 s: about
  50 %; over 16 s: 91 %). In those spells the model gives the true team about 0.45: the opponent is
  just as close, and tracking alone does not say who touched the ball. Neither reweighting the loss,
  learned transitions, nor explicit touch features moved this, so the model reports fewer, longer
  possessions than the training labels have.
- **Training labels are an approximation.** The challenge ground truth comes from corrected
  tracking by its own rules; ours come from the providers' events through the rule in §4. Where the
  definitions differ, we inherit the difference.
- **Simulated tracker.** The degradation reproduces the gap pattern, the moving dead ball and the
  ball's kinematics (velocity-change quantiles within 10 % after a 0.24 s blur, which the models
  tolerate: 0.764 / 0.784 on the blurred ball). Gaps placed where an automatic tracker tends to lose
  the ball (slow balls, near the lines) do not hurt the models. Our simulated dead ball moves more
  calmly than the challenge's.
- **Two providers for training.** Across providers we lose about 0.1 (IDSSE → Metrica); training
  the encoder on both recovers 0.01–0.02. The literature reports similar or larger drops.
- **With more time:** labelled matches from an automatic, broadcast-style tracker (the closest
  match to this domain), a ball detector trained on external wide-angle tactical footage to fill
  gaps, and per-player touch
  attribution from video for the sub-2-second possessions.

## 6. Reproduction

Python 3.12, CPU, pinned dependencies (`requirements-solution.txt`). The match data must be in
`data/` as distributed; nothing from it is in the repository.

```bash
pip install -r requirements-solution.txt
python possession/idsse.py                  # downloads the 7 IDSSE matches and their events
python possession/v5.py                     # fits the HGB on degraded IDSSE (deterministic, shared by v6)
python possession/pipeline.py --out "Barca Reborn.csv"     # v6 by default
python evaluation/validate_submission.py "Barca Reborn.csv"
```

The set-encoder weights are the only trained artefact in git (`possession/models/encoder_metrica_s*.npz`,
3 × 1.8 MB, fitted on IDSSE and Metrica only). GPU retraining is not bit-for-bit repeatable, which is
why we commit the weights. They can be retrained with `possession/experiments/encoder_train.py`
(variant `metrica`) in a separate torch environment. Inference is single-threaded float64, and the
CSV is identical from run to run: SHA-256
`a6ef47f6b3839a39b03909ceee2531bd7610820eb6943aa5774d8755a7d2d5ee`.
All experiments and their measurements are logged in `BITACORA.md`.
