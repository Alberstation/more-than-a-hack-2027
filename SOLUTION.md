# Our solution: possession/

From the provided tracking to the submission file, in four commands. Python 3.12, CPU only.

```bash
pip install -r requirements-solution.txt
python possession/idsse.py          # downloads the 7 IDSSE/DFL open matches and their events (CC-BY 4.0) to data/external/
python possession/v5.py             # degrades them, fits the HGB on control labels -> data/models/ (v5 and v6 share it)
python possession/pipeline.py --out "Barca Reborn.csv"     # v6 by default
python evaluation/validate_submission.py "Barca Reborn.csv"
```

Everything generated (tracking caches, external data, the fitted HGB, submissions) stays under
`data/` or `evaluation/results/`, both gitignored: the match data must not be redistributed. The
set-encoder weights are the one trained artefact in git (`possession/models/`, 3 seeds, 1.8 MB
each): v6's (`encoder_metrica_s*.npz`) come from IDSSE and the three Metrica sample games, v5's from
IDSSE only, and they run in NumPy, so regenerating the submission needs no GPU.

| file | what it does |
|---|---|
| `possession/atd.py` | reads the tactical ATD once into numpy arrays, cached |
| `possession/idsse.py` | downloads IDSSE/DFL and converts it to the same arrays, with H/A/D per frame |
| `possession/control.py` | control labels: a spell with no ball played by that team (official events) is a touch, handed back |
| `possession/features.py` | per-frame features, identical for both sources, no player identities |
| `possession/train.py` | degrades IDSSE to look like the challenge tracking, leave-one-match-out CV, v2 fit |
| `possession/features_v3.py` | touch, team shape and control evidence added for v3 |
| `possession/train_v3.py` | fits v3: 3 degradations of each match under 2 ball-gap patterns |
| `possession/train_v4.py` | fits v4: v3 with a dead ball that moves |
| `possession/v5.py` | v5 constants, and fits its HGB: v4 on control labels |
| `possession/encoder.py` | v5 set encoder at inference: per-player tensors and a NumPy forward pass |
| `possession/decode.py` | three-state H/A/D Viterbi with a cost per change of state |
| `possession/v6.py` | v6 constants and encoder paths: v5 with the encoder also trained on Metrica |
| `possession/pipeline.py` | v1 to v6 predictions for the challenge match; v6 is the default |

v6, the submission, is v5 with an encoder trained on two providers (IDSSE and Metrica's sample
games), decoded with costs (14, 20). v5 mixes the HGB (weight 0.25) with a set encoder (0.75): per team, a shared MLP over players
pooled by mean, max and closeness to the ball, then a dilated temporal convolution of about
+-50 s. Then one Viterbi pass per half, costs (20, 20). The encoder is trained on GPU by
`possession/experiments/encoder_train.py` in a separate torch environment, and exported by
`encoder_export.py`; retraining it on GPU is not bit-for-bit repeatable, which is why its weights
are committed. Inference is float64 on one thread, and the CSV is identical run to run.

What is fitted, and on what: only IDSSE and, for v6's encoder, Metrica's sample games (other games,
allowed by RULES.md; Metrica is credited as its README asks). The challenge match
contributes input data only -- its tracking, Smart Tagging kickoffs, and the length of its ball
gaps for the degradation. Its public five minutes are used for nothing but a format check.

The plan, the rules checklist and the references are in `docs/plan/plan_de_mejora.tex`.
