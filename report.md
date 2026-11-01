# Repeat first: content-based filtering and matrix factorization on SWEET.TV

- **Student:** Egor Antonov
- **Email:** eaantonov@edu.hse.ru
- **Date:** 2026-11-01
- **Assignment:** Midterm (FKN) &mdash; TV program recommendation
- **Competition:** [sweet.tv &mdash; TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender)
- **GitHub:** [github.com/xcelsior11/LLM4RecSys](https://github.com/xcelsior11/LLM4RecSys)
- **Kaggle:** username `sentineln1`. Designated final submission: `final.csv` (popularity). Returned **private MAP@5 = 0.39731**, public 0.41355.

![Kaggle submissions page: final.csv private 0.39731, public 0.41355](kaggle_submissions.png)

## Abstract

This work builds a Top-5 TV-programme list for each SWEET.TV test user. Viewing logs are joined to the schedule by channel and overlapping time. Rows with `tv_show_id = 0` are dropped. Three methods share one temporal split (train weeks 11–20, validation weeks 21–30) and one MAP@5 protocol: a popularity reference, content-based filtering, and matrix factorization. Popularity reaches validation MAP@5 **0.525**. CBF reaches **0.174**. A small SGD factor model reaches **0.008**. The Kaggle file is the popularity model retrained on weeks 11–30. Its private MAP@5 is **0.39731**. For SWEET.TV I would ship the repeat-popularity list and keep CBF only as a cold-start explanation, not as the main row.

**Index terms:** content-based filtering, matrix factorization, MAP@5, TV recommendation, SWEET.TV

## 1. Introduction

The log records which channel a user was on, not which programme they finished. Only about 11% of schedule rows have a programme id, so the first step is a join. Linear TV is also habitual: news and series come back every day. That makes a user’s own history a strong predictor.

I implement the two required methods &mdash; content-based filtering and matrix factorization &mdash; plus the required popularity baseline, all on the same split. The code uses pandas to read the CSVs, NumPy only for the time-overlap join, and ordinary Python dicts and loops for the recommenders.

## 2. Related work

Pazzani and Billsus describe a content-based profile as a combination of item attributes [1]. Koren, Bell and Volinsky review matrix factorization for ratings [2]. Cremonesi, Koren and Turrin show that a rating-style model can lose to popularity on a top-N task [4]. That is what happens here. The official metric is MAP@5 [5]. A 2020 write-up of this same competition already reported that repeating a user’s own finished programmes, restricted to the future grid, scores about 0.41 public MAP@5 [6]. I treat that as related work, not as a number I computed.

## 3. Method

### 3.1 Join, missing ids, watch rule

View columns: `user_id`, `vsetv_id`, `start_time`, `stop_time`, `duraton` (session length in seconds). Schedule columns include `channel_id`, `start_time`, `duration`, `tv_show_id`, title, category and three genres.

**Join.** A view matches a slot when the channel is the same (`vsetv_id = channel_id`) and the two intervals overlap by at least one second:

`overlap = min(view_stop, show_stop) - max(view_start, show_start)`.

**Missing ids.** `tv_show_id = 0` is unlabelled airtime. Those rows are dropped. We do not invent an id from the title. Labelled share of the three schedule files: 10.76%, 11.31%, 11.18%.

**Watch.** `duraton` is not the programme length. A match counts as a watch when `overlap >= 0.5 * duration`. This is the same 50% cut used in [6]. It gives 177,849 train pairs and 197,428 validation pairs.

**User ids.** About half of the raw ids are larger than `2^63 - 1`. Storing them as `int64` wraps them to negative numbers. Every user id is kept as a string from the CSV through the submission file.

### 3.2 Split, pool, repeats, cold start

| Window | What we use | Role |
|---|---|---|
| Weeks 11–20 | views joined to labelled `export_arh_11-20` | train, for model selection |
| Weeks 21–30 | views joined to labelled `export_arh_21-30` | validation |
| Weeks 31–42 | labelled `export_arh_31-42` only | Kaggle candidate pool |

Validation is never used to fit a model that is then scored on validation. The Kaggle models are fit again on weeks 11–30, which are all in the past relative to the test grid. There are no test-window views in the files.

The candidate pool is the set of labelled programme ids in the target window (6,363 on validation, 6,636 on test). A method may not emit an id outside that set.

Already-watched programmes may be recommended. The same rule is used for every method. 15 of 1,260 test users have no train pair; they all receive the same global popularity fill.

Each method returns five ids. Unused slots are filled from global popularity.

### 3.3 Models

**Popularity (reference, not one of the two required methods).** For each user, sort their own train programmes by watch count and keep those that are still in the pool. Fill the rest from the global watch-count list. 0.2 s on validation.

**Content-based filtering.** Each programme is a small set of tags: category plus up to three genres. The user profile is a count of those tags over the programmes they watched. The score of a candidate is cosine between the profile and the candidate’s tags. 15 s.

**Matrix factorization.** Each user and each programme gets a 16-dimensional vector. On every train pair we take the target to be 1 and take one SGD step on the squared error of the dot product, with a small L2 penalty (8 epochs, learning rate 0.05, seed 42). This is the same update idea as HW5, without a rating scale. 22 s.

Environment: Python 3.11.18, pandas 3.1.0, NumPy 2.4.6.

## 4. Experiments

### 4.1 Shared validation

4,721 users with at least one validation watch. Same pool, same `K = 5`.

`AP@5 = (sum of precision at each hit) / min(number of relevant items, 5)`. MAP@5 is the mean of AP@5.

| Method | MAP@5 | Recall@5 | Seconds |
|---|---:|---:|---:|
| Popularity | **0.52521** | **0.12182** | 0.2 |
| CBF | 0.17388 | 0.05856 | 14.9 |
| MF | 0.00757 | 0.00315 | 21.7 |

Kaggle, popularity retrained on weeks 11–30: public **0.41355**, private **0.39731** (ref 57867800). CBF was also submitted as a check: public 0.26513, private 0.27013 (ref 57867824; that file was the earlier TF-IDF variant). The designated file is `submissions/final.csv`.

### 4.2 Hand-checked MAP@5

User `13571199046623737884`, popularity list `[2400480, 240081, 20088, 12001682, 2400508]`. Hits at ranks 1, 2 and 5 (`2400480`, `240081`, `2400508`).

- P@1 = 1/1 = 1
- P@2 = 2/2 = 1
- P@5 = 3/5 = 0.6
- Sum = 2.6
- Denominator = min(46, 5) = 5
- AP@5 = 2.6 / 5 = **0.52**

The code returns 0.52 on the same user and list.

### 4.3 Two users

The same two test users are used for every method.

**User A** `13571199046623737884` &mdash; 1,932 train watches on 251 programmes, mostly news (ТСН, Факты, Сегодня). A heavy linear-TV subscriber.

**User B** `8766761403437974146` &mdash; one train watch (`Отдел 44`, series). A near-cold user.

Both have validation watches, so the lists can be compared with held-out behaviour.

| Rank | A popularity | A CBF | A MF |
|---|---|---|---|
| 1 | 2400480 ТСН. Спецпроект. (Инфо) | 5900347 Деньги. (Инфо) | 10003122 Моя чужая жизнь (Сериалы) |
| 2 | 240081 Факты. День. (Инфо) | 2400480 ТСН. Спецпроект. (Инфо) | 90075475 Кон-Тики (Фильмы) |
| 3 | 20088 Завтрак с 1+1 (Развлечения) | 2500261 Подробности недели. (Инфо) | 90072464 Днепровский рубеж (Фильмы) |
| 4 | 12001682 Губка Боб (Детям) | 240081 Факты. День. (Инфо) | 90076041 Ренуар (Фильмы) |
| 5 | 2400508 Сегодня. Итоги (Инфо) | 2400467 Викна-Новости. (Инфо) | 90011459 Побег (Фильмы) |

A’s validation truth includes ТСН, Факты and Сегодня. Popularity puts three of those in the Top-5. CBF stays inside **Инфо** and still recovers ТСН and Факты. MF leaves the news habit and returns films. That is “new”, not evidence that A wanted those films.

| Rank | B popularity | B CBF | B MF |
|---|---|---|---|
| 1 | 10003014 Отдел 44 (Сериалы) | 10001051 Иван Подушкин (Сериалы) | 90047709 Другие (Фильмы) |
| 2 | 2400480 ТСН (Инфо) | 10001438 Морозов (Сериалы) | 90069390 Загадки Сфинкса (Фильмы) |
| 3 | 240081 Факты. День. (Инфо) | 10001457 Супруги (Сериалы) | 120078334 Тинтин (Детям) |
| 4 | 12001682 Губка Боб (Детям) | 10001533 Капитан Черняев (Сериалы) | 5300397 Сказка с папой (Другое) |
| 5 | 20088 Завтрак с 1+1 (Развлечения) | 10001610 Иван Подушкин 2 (Сериалы) | 120078223 Лесной патруль (Детям) |

B’s one history item is rank 1 for popularity, then the global news/kids head. CBF stays in **Сериалы**, which matches B’s validation series (`Морской патруль`, `Спартак`, `Двойная жизнь`) better than SpongeBob. Because almost every series has the same category tag, CBF cannot put `Отдел 44` first: all series look alike. MF does not recover the series at all.

## 5. Discussion

### 5.1 Failure: `int64` user ids

**Detect.** A first pass stored `user_id` as `int64`. The two-user picker printed a negative id. 2,918,899 view rows and 655 of 1,260 test users have ids larger than `2^63 - 1`.

**Cause.** `int64` wraps those values. Train and validation stayed consistent with each other, so MAP@5 on the hold-out did not move. The Kaggle template would not have matched 655 users.

**Fix.** Every user key is a decimal string.

**Confirm.** The designated `final.csv` has the same 1,260 user ids as the template, in the same order, and five ids on every row. It scored 0.41355 / 0.39731, not the template’s 0.00000.

### 5.2 Failure: SGD matrix factorization

The factor model only sees positive pairs and the target is always 1. After a few epochs the vectors do not rank the user’s own news above random films. Validation MAP@5 is 0.0076, below a constant list. This is a modelling mistake, not a data bug: without negative examples the update has nothing to push down. I left the simple SGD in the code so the comparison is honest. I did not switch to a library ALS just to raise this number.

### 5.3 Which strategy for SWEET.TV?

**Ship per-user repeat popularity. Use CBF only to fill cold-start users and to explain “more news like this”. Do not ship this MF as the main row.**

On the shared hold-out, popularity is 0.525 MAP@5 against 0.174 (CBF) and 0.008 (MF). The Kaggle private scores keep that order for popularity vs CBF (0.397 vs 0.270). User A’s news habit is recovered by popularity and CBF and missed by MF. User B is the one place CBF is more readable: it stays in series instead of inserting the global kids/news head.

Cost. Popularity is a few dicts (0.2 s). CBF is 15 s. MF is 22 s and still loses.

For a heavy news viewer, repeating last week’s newscasts is the product. An unfamiliar MF film is not automatically a useful discovery [4]. For a one-title user, “more series” is a better first message than SpongeBob, even when MAP@5 on the full cohort still prefers popularity.

Limits that the results actually support:

1. About 89% of airtime has no programme id, so most viewing never becomes an item.
2. The log is this platform only. Absence is not “never watched”.
3. The 50% duration cut ignores short news drop-ins.
4. Validation is the previous block of the same grid, not a live week.
5. CBF tags are coarse: all series share one category, so ties are weak.
6. No A/B test and no revenue claim.

## 6. AI usage disclosure

- **Tools:** Cursor agent; Python 3.11.18, pandas 3.1.0, NumPy 2.4.6; Kaggle CLI.
- **How AI was used:** writing and debugging code; drafting the report.
- **What I verified afterwards:** the three validation MAP@5 values in `artifacts/val_metrics.csv` (popularity 0.52521, CBF 0.17388, MF 0.00757); the hand AP@5 = 2.6 / 5 = 0.52 in §4.2; the Kaggle screenshot for `final.csv` (private 0.39731).
- **Trusted without a second implementation:** the exact SGD step sizes in the factor model.
- **Session:** `egor_antonov_midterm_session.json`.

## References

[1] M. J. Pazzani and D. Billsus, “Content-Based Recommendation Systems,” in *The Adaptive Web*, LNCS 4321, Springer, 2007, pp. 325–341, doi: 10.1007/978-3-540-72079-9_10. [Accessed: 2026-11-01]

[2] Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” *Computer*, vol. 42, no. 8, pp. 30–37, 2009, doi: 10.1109/MC.2009.263. [Accessed: 2026-11-01]

[3] Y. Koren, “Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model,” in *Proc. ACM KDD*, 2008, pp. 426–434, doi: 10.1145/1401890.1401944. [Accessed: 2026-11-01]

[4] P. Cremonesi, Y. Koren, and R. Turrin, “Performance of Recommender Algorithms on Top-N Recommendation Tasks,” in *Proc. ACM RecSys*, 2010, pp. 39–46, doi: 10.1145/1864708.1864721. [Accessed: 2026-11-01]

[5] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge Univ. Press, 2008, ch. 8. [Online]. Available: https://nlp.stanford.edu/IR-book/ [Accessed: 2026-11-01]

[6] N. Melentev, “Two-layered recommender system methodology: a prize-winning solution,” *Towards Data Science*, 20 Mar. 2021. [Online]. Available: https://towardsdatascience.com/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156/ [Accessed: 2026-11-01]

[7] “sweet.tv &mdash; TV program recommender,” Kaggle competition 24201. [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender [Accessed: 2026-11-01]
