"""Join views to labelled schedule slots. Write CSV tables."""

import csv
import json
import time

import numpy as np
import pandas as pd

from config import (
    ART_DIR,
    MIN_OVERLAP_SEC,
    SCHED_TEST,
    SCHED_TRAIN,
    SCHED_VAL,
    TEMPLATE,
    VIEW_LOG,
    WATCH_FRAC,
)


def load_schedule(path):
    df = pd.read_csv(path, low_memory=False)
    df["tv_show_id"] = pd.to_numeric(df["tv_show_id"], errors="coerce").fillna(0).astype(int)
    n_all = len(df)
    labeled = df[df["tv_show_id"] != 0].copy()
    labeled["start_dt"] = pd.to_datetime(labeled["start_time"], format="%d.%m.%Y %H:%M:%S")
    labeled["duration"] = pd.to_numeric(labeled["duration"], errors="coerce").fillna(0).astype(int)
    labeled["stop_dt"] = labeled["start_dt"] + pd.to_timedelta(labeled["duration"], unit="s")
    labeled["channel_id"] = pd.to_numeric(labeled["channel_id"], errors="coerce").astype(int)
    return labeled, n_all


def load_views(path):
    df = pd.read_csv(path)
    df["user_id"] = df["user_id"].astype(str)
    df["vsetv_id"] = pd.to_numeric(df["vsetv_id"], errors="coerce").astype(int)
    df["start_dt"] = pd.to_datetime(df["start_time"])
    df["stop_dt"] = pd.to_datetime(df["stop_time"])
    return df


def join_views_to_schedule(views, sched):
    """Same channel and overlapping time. Returns list of (user, show, overlap, duration)."""
    views_by_ch = {}
    for ch, group in views.groupby("vsetv_id", sort=False):
        views_by_ch[int(ch)] = group

    matches = []
    block = 3000
    for ch, shows in sched.groupby("channel_id", sort=False):
        group = views_by_ch.get(int(ch))
        if group is None:
            continue
        v_start = group["start_dt"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        v_stop = group["stop_dt"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        v_user = group["user_id"].to_numpy()
        s_start = shows["start_dt"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        s_stop = shows["stop_dt"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        s_id = shows["tv_show_id"].to_numpy()
        s_dur = shows["duration"].to_numpy()

        n_views = len(v_start)
        for begin in range(0, n_views, block):
            end = min(begin + block, n_views)
            overlap = np.minimum(v_stop[begin:end][:, None], s_stop) - np.maximum(
                v_start[begin:end][:, None], s_start
            )
            vi, si = np.where(overlap >= MIN_OVERLAP_SEC)
            for a, b in zip(vi.tolist(), si.tolist()):
                matches.append(
                    (
                        str(v_user[begin + a]),
                        int(s_id[b]),
                        int(overlap[a, b]),
                        int(s_dur[b]),
                    )
                )
    return matches


def aggregate(matches, watch_frac):
    """Keep watches with overlap >= watch_frac * duration. Count per user-programme."""
    counts = {}
    seconds = {}
    for user, item, overlap, duration in matches:
        if overlap < watch_frac * duration:
            continue
        key = (user, item)
        counts[key] = counts.get(key, 0) + 1
        seconds[key] = seconds.get(key, 0) + overlap
    rows = []
    for (user, item), n in counts.items():
        rows.append((user, item, n, seconds[(user, item)]))
    return rows


def write_pairs(path, rows, header):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def catalog_rows(frames):
    cols = [
        "tv_show_id",
        "tv_show_title",
        "tv_show_category",
        "tv_show_genre_1",
        "tv_show_genre_2",
        "tv_show_genre_3",
        "channel_title",
    ]
    seen = {}
    for df in frames:
        for row in df[cols].itertuples(index=False):
            seen[int(row[0])] = row
    out = []
    for show_id in seen:
        r = seen[show_id]
        out.append(
            {
                "tv_show_id": int(r[0]),
                "tv_show_title": r[1],
                "tv_show_category": r[2],
                "tv_show_genre_1": r[3],
                "tv_show_genre_2": r[4],
                "tv_show_genre_3": r[5],
                "channel_title": r[6],
            }
        )
    return out


def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading schedules…")
    train_s, n_train_s = load_schedule(SCHED_TRAIN)
    val_s, n_val_s = load_schedule(SCHED_VAL)
    test_s, n_test_s = load_schedule(SCHED_TEST)
    print("Loading views…")
    views = load_views(VIEW_LOG)
    test_users = pd.read_csv(TEMPLATE, dtype={"user_id": str})["user_id"].astype(str).tolist()

    print("Joining train window…")
    t1 = time.time()
    train_raw = join_views_to_schedule(views, train_s)
    print("  train overlaps:", len(train_raw), "in", round(time.time() - t1, 1), "s")
    print("Joining val window…")
    t1 = time.time()
    val_raw = join_views_to_schedule(views, val_s)
    print("  val overlaps:", len(val_raw), "in", round(time.time() - t1, 1), "s")

    train_agg = aggregate(train_raw, WATCH_FRAC)
    val_agg = aggregate(val_raw, WATCH_FRAC)
    write_pairs(ART_DIR / "train_interactions.csv", train_agg, ["user_id", "tv_show_id", "n_watches", "watch_sec"])
    write_pairs(ART_DIR / "val_interactions.csv", val_agg, ["user_id", "tv_show_id", "n_watches", "watch_sec"])

    catalog = catalog_rows([train_s, val_s, test_s])
    with (ART_DIR / "item_catalog.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tv_show_id",
                "tv_show_title",
                "tv_show_category",
                "tv_show_genre_1",
                "tv_show_genre_2",
                "tv_show_genre_3",
                "channel_title",
            ],
        )
        w.writeheader()
        w.writerows(catalog)

    pool_val = sorted(int(x) for x in val_s["tv_show_id"].unique())
    pool_test = sorted(int(x) for x in test_s["tv_show_id"].unique())
    write_pairs(ART_DIR / "pool_val.csv", [(i,) for i in pool_val], ["tv_show_id"])
    write_pairs(ART_DIR / "pool_test.csv", [(i,) for i in pool_test], ["tv_show_id"])

    train_users = {u for u, _i, _n, _s in train_agg}
    val_users = {u for u, _i, _n, _s in val_agg}
    stats = {
        "watch_frac": WATCH_FRAC,
        "join_rule": "same channel and interval overlap >= 1s",
        "missing_id_rule": "drop tv_show_id == 0",
        "n_view_rows": int(len(views)),
        "n_view_users": int(views["user_id"].nunique()),
        "schedule_rows": {"train": n_train_s, "val": n_val_s, "test": n_test_s},
        "labeled_slots": {"train": int(len(train_s)), "val": int(len(val_s)), "test": int(len(test_s))},
        "labeled_frac": {
            "train": len(train_s) / n_train_s,
            "val": len(val_s) / n_val_s,
            "test": len(test_s) / n_test_s,
        },
        "n_overlaps": {"train": len(train_raw), "val": len(val_raw)},
        "n_positive_pairs": {"train": len(train_agg), "val": len(val_agg)},
        "n_positive_users": {"train": len(train_users), "val": len(val_users)},
        "n_items": {
            "catalog": len(catalog),
            "val_pool": len(pool_val),
            "test_pool": len(pool_test),
        },
        "n_test_users": len(test_users),
        "n_test_users_with_train_history": len(train_users.intersection(test_users)),
        "elapsed_sec": time.time() - t0,
    }
    (ART_DIR / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("Done in", round(stats["elapsed_sec"], 1), "s")


if __name__ == "__main__":
    main()
