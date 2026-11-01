"""MAP@5 and Recall@5. Same functions for every method."""


def average_precision_at_k(pred, relevant, k=5):
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(pred[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def recall_at_k(pred, relevant, k=5):
    if not relevant:
        return 0.0
    hit = 0
    pred_set = set(pred[:k])
    for item in relevant:
        if item in pred_set:
            hit += 1
    return hit / len(relevant)


def evaluate_map(recs, truth, k=5):
    aps = []
    recalls = []
    for user, relevant in truth.items():
        if not relevant:
            continue
        pred = recs.get(user, [])
        aps.append(average_precision_at_k(pred, relevant, k))
        recalls.append(recall_at_k(pred, relevant, k))
    n = len(aps)
    return {
        "n_eval_users": n,
        "map@5": (sum(aps) / n) if n else 0.0,
        "recall@5": (sum(recalls) / n) if n else 0.0,
    }


def truth_from_rows(rows):
    """rows: list of (user_id, tv_show_id)."""
    truth = {}
    for user, item in rows:
        if user not in truth:
            truth[user] = set()
        truth[user].add(int(item))
    return truth


def write_submission(path, user_ids, recs, k=5):
    lines = ["user_id,tv_show_id"]
    for user in user_ids:
        ids = [str(int(x)) for x in recs.get(user, [])[:k]]
        while len(ids) < k:
            ids.append("0")
        lines.append(user + "," + " ".join(ids))
    path.write_text("\n".join(lines) + "\n")
