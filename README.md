# SWEET.TV midterm &mdash; TV program recommendation

HSE LLM4Rec midterm (FKN, 2026-11-01).
Kaggle: [sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender).

Student: Egor Antonov (`eaantonov@edu.hse.ru`). Kaggle name: `sentineln1`.

## Designated Kaggle file

`submissions/final.csv` &mdash; per-user repeat popularity, filled with global popularity.

- Submission ref: **57867800**
- Public MAP@5: **0.41355**
- Private MAP@5: **0.39731**

## How to run

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
# put the five Kaggle CSVs in data/
.venv/bin/python src/prepare.py
.venv/bin/python src/run.py
```

`prepare.py` joins views to labelled programmes and writes CSVs in `artifacts/`.
`run.py` scores popularity, content-based filtering and matrix factorization on the same split, then writes `submissions/`.

## Rules used in the code

- Join: same channel and overlapping time (at least 1 second).
- Rows with `tv_show_id = 0` are dropped.
- A match is a watch if the overlap is at least half of the programme duration.
- Already-watched programmes may be recommended.
- Users with no history get the global popularity list.
- User ids are stored as strings (they do not fit in `int64`).
