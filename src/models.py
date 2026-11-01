"""Three recommenders: popularity, content-based filtering, matrix factorization."""

import math
import random

import numpy as np

from config import LEARNING_RATE, N_EPOCHS, N_FACTORS, N_RECS, REG, SEED


def take_top(ranked, fallback, k=N_RECS):
    """Keep the first k unique ids, filling from fallback if needed."""
    out = []
    seen = set()
    for item in ranked + fallback:
        item = int(item)
        if item == 0 or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) == k:
            break
    return out


def global_popularity(train_rows, pool):
    """train_rows: list of (user, item, n_watches)."""
    counts = {}
    for _user, item, n in train_rows:
        item = int(item)
        if item not in pool:
            continue
        counts[item] = counts.get(item, 0) + n
    items = list(counts.keys())
    items.sort(key=lambda i: -counts[i])
    return items


def recommend_popularity(users, train_rows, pool, k=N_RECS):
    fallback = global_popularity(train_rows, pool)
    by_user = {}
    for user, item, n in train_rows:
        item = int(item)
        if item not in pool:
            continue
        if user not in by_user:
            by_user[user] = {}
        by_user[user][item] = by_user[user].get(item, 0) + n

    recs = {}
    for user in users:
        hist = by_user.get(user, {})
        ranked = list(hist.keys())
        ranked.sort(key=lambda i: -hist[i])
        recs[user] = take_top(ranked, fallback, k)
    return recs


def _clean(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("", "nan", "<na>", "none"):
        return ""
    return text


def item_tags(row):
    """A small set of content tags: category + three genres."""
    tags = []
    for key in ("tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
        text = _clean(row.get(key))
        if text:
            tags.append(text.lower())
    return tags


def recommend_cbf(users, train_rows, catalog, pool, fallback, k=N_RECS):
    """User profile = sum of tag counts from programmes they watched. Score = cosine."""
    tags_of = {}
    for row in catalog:
        show_id = int(row["tv_show_id"])
        tags_of[show_id] = item_tags(row)

    # user -> {tag: weight}
    profiles = {}
    for user, item, n in train_rows:
        item = int(item)
        if item not in tags_of:
            continue
        if user not in profiles:
            profiles[user] = {}
        for tag in tags_of[item]:
            profiles[user][tag] = profiles[user].get(tag, 0.0) + float(n)

    # tag -> programmes in the pool that have this tag
    tag_to_items = {}
    item_norm = {}
    for item in pool:
        tags = tags_of.get(item, [])
        if not tags:
            continue
        item_norm[item] = math.sqrt(len(tags))
        for tag in tags:
            if tag not in tag_to_items:
                tag_to_items[tag] = []
            tag_to_items[tag].append(item)

    recs = {}
    for user in users:
        profile = profiles.get(user)
        if not profile:
            recs[user] = take_top([], fallback, k)
            continue
        p_norm = math.sqrt(sum(w * w for w in profile.values()))
        dots = {}
        for tag, w in profile.items():
            for item in tag_to_items.get(tag, []):
                dots[item] = dots.get(item, 0.0) + w
        scored = []
        for item, dot in dots.items():
            scored.append((item, dot / (p_norm * item_norm[item])))
        scored.sort(key=lambda x: -x[1])
        ranked = [item for item, _score in scored]
        recs[user] = take_top(ranked, fallback, k)
    return recs


def recommend_mf(users, train_rows, pool, fallback, k=N_RECS):
    """Dot-product MF trained with SGD on implicit watches (target = 1)."""
    user_ids = []
    item_ids = []
    seen_u = set()
    seen_i = set()
    for user, item, _n in train_rows:
        if user not in seen_u:
            seen_u.add(user)
            user_ids.append(user)
        item = int(item)
        if item not in seen_i:
            seen_i.add(item)
            item_ids.append(item)
    for user in users:
        if user not in seen_u:
            seen_u.add(user)
            user_ids.append(user)
    for item in pool:
        item = int(item)
        if item not in seen_i:
            seen_i.add(item)
            item_ids.append(item)

    u_index = {user: i for i, user in enumerate(user_ids)}
    i_index = {item: i for i, item in enumerate(item_ids)}

    random.seed(SEED)
    n_u = len(user_ids)
    n_i = len(item_ids)
    P = [[random.uniform(-0.1, 0.1) for _ in range(N_FACTORS)] for _ in range(n_u)]
    Q = [[random.uniform(-0.1, 0.1) for _ in range(N_FACTORS)] for _ in range(n_i)]

    pairs = []
    watched = {}
    for user, item, _n in train_rows:
        ui = u_index[user]
        ii = i_index[int(item)]
        pairs.append((ui, ii))
        if ui not in watched:
            watched[ui] = set()
        watched[ui].add(ii)

    for _epoch in range(N_EPOCHS):
        random.shuffle(pairs)
        for ui, ii in pairs:
            pred = 0.0
            for f in range(N_FACTORS):
                pred += P[ui][f] * Q[ii][f]
            err = 1.0 - pred
            for f in range(N_FACTORS):
                pu = P[ui][f]
                qi = Q[ii][f]
                P[ui][f] += LEARNING_RATE * (err * qi - REG * pu)
                Q[ii][f] += LEARNING_RATE * (err * pu - REG * qi)

    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    recs = {}
    pool_items = [item for item in pool if item in i_index]
    pool_idx = [i_index[item] for item in pool_items]
    Q_pool = Q[pool_idx]
    for user in users:
        ui = u_index[user]
        if ui not in watched:
            recs[user] = take_top([], fallback, k)
            continue
        scores = Q_pool.dot(P[ui])
        order = list(range(len(pool_items)))
        order.sort(key=lambda j: -scores[j])
        ranked = [pool_items[j] for j in order]
        recs[user] = take_top(ranked, fallback, k)
    return recs
