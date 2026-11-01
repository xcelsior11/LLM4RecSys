"""Evaluate the three methods on the same split and write Kaggle files."""

import csv
import json
import time

from config import ART_DIR, N_RECS, SEED, SUB_DIR, TEMPLATE
from evaluate import average_precision_at_k, evaluate_map, truth_from_rows, write_submission
from models import global_popularity, recommend_cbf, recommend_mf, recommend_popularity


def read_pairs(path):
    rows = []
    with path.open(newline="") as f:
        for rec in csv.DictReader(f):
            rows.append((rec["user_id"], int(rec["tv_show_id"]), int(float(rec["n_watches"]))))
    return rows


def read_catalog(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_pool(path):
    ids = set()
    with path.open(newline="") as f:
        for rec in csv.DictReader(f):
            ids.add(int(rec["tv_show_id"]))
    return ids


def read_users(path):
    users = []
    with path.open(newline="") as f:
        for rec in csv.DictReader(f):
            users.append(rec["user_id"])
    return users


def lookup(ids, catalog):
    by_id = {int(r["tv_show_id"]): r for r in catalog}
    out = []
    for i in ids:
        row = by_id.get(int(i), {})
        out.append(
            {
                "tv_show_id": int(i),
                "title": row.get("tv_show_title", "?"),
                "category": row.get("tv_show_category", "?"),
            }
        )
    return out


def pick_two_users(train_rows, val_users, test_users):
    counts = {}
    nprog = {}
    for user, item, n in train_rows:
        counts[user] = counts.get(user, 0) + n
        if user not in nprog:
            nprog[user] = set()
        nprog[user].add(item)
    candidates = [u for u in test_users if u in counts and u in val_users]
    candidates.sort(key=lambda u: -counts[u])
    heavy = candidates[0]
    light = candidates[0]
    best = counts[candidates[0]]
    for u in candidates:
        if counts[u] < best:
            best = counts[u]
            light = u
    return heavy, light


def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train = read_pairs(ART_DIR / "train_interactions.csv")
    val = read_pairs(ART_DIR / "val_interactions.csv")
    catalog = read_catalog(ART_DIR / "item_catalog.csv")
    pool_val = read_pool(ART_DIR / "pool_val.csv")
    pool_test = read_pool(ART_DIR / "pool_test.csv")
    test_users = read_users(TEMPLATE)
    val_users = sorted({u for u, _i, _n in val})
    truth = truth_from_rows([(u, i) for u, i, _n in val])

    fallback_val = global_popularity(train, pool_val)
    timings = {}
    val_recs = {}
    metrics = {}

    t0 = time.time()
    val_recs["popularity"] = recommend_popularity(val_users, train, pool_val)
    timings["popularity"] = time.time() - t0
    metrics["popularity"] = evaluate_map(val_recs["popularity"], truth)

    t0 = time.time()
    val_recs["cbf"] = recommend_cbf(val_users, train, catalog, pool_val, fallback_val)
    timings["cbf"] = time.time() - t0
    metrics["cbf"] = evaluate_map(val_recs["cbf"], truth)

    t0 = time.time()
    val_recs["mf"] = recommend_mf(val_users, train, pool_val, fallback_val)
    timings["mf"] = time.time() - t0
    metrics["mf"] = evaluate_map(val_recs["mf"], truth)

    print("Validation")
    for name in ("popularity", "cbf", "mf"):
        m = metrics[name]
        print(
            " ",
            name,
            "MAP@5=%.5f" % m["map@5"],
            "Recall@5=%.5f" % m["recall@5"],
            "t=%.1fs" % timings[name],
        )

    best = "popularity"
    for name in metrics:
        if metrics[name]["map@5"] > metrics[best]["map@5"]:
            best = name
    print("best on val:", best)

    u_a, u_b = pick_two_users(train, set(val_users), test_users)
    two = {}
    for uid, tag in ((u_a, "A_heavy"), (u_b, "B_light")):
        hist = [(i, n) for u, i, n in train if u == uid]
        hist.sort(key=lambda x: -x[1])
        two[tag] = {
            "user_id": uid,
            "n_train_watches": sum(n for _i, n in hist),
            "n_train_programs": len(hist),
            "train_top": lookup([i for i, _n in hist[:8]], catalog),
            "val_truth": lookup(sorted(truth.get(uid, [])), catalog),
            "methods": {name: lookup(val_recs[name].get(uid, []), catalog) for name in val_recs},
        }

    pred = val_recs[best].get(u_a, [])
    relevant = truth.get(u_a, set())
    steps = []
    running = []
    hits = 0
    for rank, item in enumerate(pred[:N_RECS], start=1):
        is_hit = item in relevant
        if is_hit:
            hits += 1
            running.append(hits / rank)
        steps.append({"rank": rank, "item": int(item), "hit": is_hit})
    denom = min(len(relevant), N_RECS) if relevant else 0
    hand = {
        "user_id": u_a,
        "pred": [int(x) for x in pred[:N_RECS]],
        "steps": steps,
        "sum_precision_at_hits": sum(running),
        "denominator": denom,
        "ap@5": (sum(running) / denom) if denom else 0.0,
        "code_ap@5": average_precision_at_k(pred, relevant, N_RECS),
    }

    # Kaggle models use all history before the test grid (weeks 11-30).
    full = {}
    for user, item, n in train + val:
        key = (user, item)
        full[key] = full.get(key, 0) + n
    full_rows = [(u, i, n) for (u, i), n in full.items()]
    fallback_test = global_popularity(full_rows, pool_test)

    test_recs = {}
    t0 = time.time()
    test_recs["popularity"] = recommend_popularity(test_users, full_rows, pool_test)
    timings["popularity_test"] = time.time() - t0
    t0 = time.time()
    test_recs["cbf"] = recommend_cbf(test_users, full_rows, catalog, pool_test, fallback_test)
    timings["cbf_test"] = time.time() - t0
    t0 = time.time()
    test_recs["mf"] = recommend_mf(test_users, full_rows, pool_test, fallback_test)
    timings["mf_test"] = time.time() - t0

    for name, recmap in test_recs.items():
        write_submission(SUB_DIR / (name + ".csv"), test_users, recmap)
    write_submission(SUB_DIR / "final.csv", test_users, test_recs[best])

    payload = {
        "seed": SEED,
        "best_method": best,
        "validation": metrics,
        "timings_sec": timings,
        "two_users": two,
        "hand_map_check": hand,
        "n_train_pairs": len(train),
        "n_val_pairs": len(val),
        "n_val_users": len(val_users),
        "n_test_users": len(test_users),
        "pool_val": len(pool_val),
        "pool_test": len(pool_test),
    }
    (ART_DIR / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with (ART_DIR / "val_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_eval_users", "map@5", "recall@5", "seconds"])
        for name in ("popularity", "cbf", "mf"):
            m = metrics[name]
            w.writerow([name, m["n_eval_users"], m["map@5"], m["recall@5"], timings[name]])
    print("Wrote artifacts. best=", best)


if __name__ == "__main__":
    main()
