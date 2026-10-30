# Our solution: possession/

From the provided tracking to the submission file, in four commands. Python 3.12.

```bash
pip install -r requirements-solution.txt
python possession/idsse.py          # downloads the 7 IDSSE/DFL open matches (CC-BY 4.0) to data/external/
python possession/train.py          # degrades them, cross-validates, fits the model -> data/models/
python possession/pipeline.py --out <team_name>.csv
python evaluation/validate_submission.py <team_name>.csv
```

Everything generated (tracking caches, external data, the fitted model, submissions) stays under
`data/` or `evaluation/results/`, both gitignored: the match data must not be redistributed.

| file | what it does |
|---|---|
| `possession/atd.py` | reads the tactical ATD once into numpy arrays, cached |
| `possession/idsse.py` | downloads IDSSE/DFL and converts it to the same arrays, with H/A/D per frame |
| `possession/features.py` | per-frame features, identical for both sources, no player identities |
| `possession/train.py` | degrades IDSSE to look like the challenge tracking, leave-one-match-out CV, final fit |
| `possession/decode.py` | three-state H/A/D Viterbi with a cost per change of state |
| `possession/pipeline.py` | v1 (hand-set evidence) and v2 (learned) predictions for the challenge match |

What is fitted, and on what: only IDSSE (other games, allowed by RULES.md). The challenge match
contributes input data only -- its tracking, Smart Tagging kickoffs, and the length of its ball
gaps for the degradation. Its public five minutes are used for nothing but a format check.

The plan, the rules checklist and the references are in `docs/plan/plan_de_mejora.tex`.
