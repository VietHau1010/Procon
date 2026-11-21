"""Small three-head ridge value model, fitted without external dependencies.

Predict future new types, future daily types and future servings separately;
never learn a scalar that could silently trade L for Q.
"""

import math
from pathlib import Path

from ..storage import atomic_json, digest, json_bytes, read_json, writer_lock
from .data import read_groups, verify
from .features import FEATURES


def _rows(groups):
    return [row for group in groups for row in group["rows"]]


def _check_row(row):
    x, y, bounds = row["features"], row["future"], row["future_bounds"]
    if (len(x) != len(FEATURES) or len(y) != 3 or len(bounds) != 3
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x+y+bounds)
            or any(b < 0 or not 0 <= v <= b for v, b in zip(y, bounds))):
        raise ValueError("Sample value không đúng feature/label contract")


def _solve(matrix, target):
    """Gaussian elimination with pivoting; matrices are only ~28x28."""
    n = len(target)
    a = [list(row) + [target[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(a[i][column]))
        if abs(a[pivot][column]) < 1e-12:
            raise ValueError("Ridge matrix suy biến")
        a[column], a[pivot] = a[pivot], a[column]
        factor = a[column][column]
        a[column] = [v/factor for v in a[column]]
        for i in range(n):
            if i == column:
                continue
            factor = a[i][column]
            for j in range(column, n+1):
                a[i][j] -= factor*a[column][j]
    return [a[i][-1] for i in range(n)]


def fit_ridge(groups, regularization):
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization phải dương và hữu hạn")
    rows = _rows(groups)
    if not rows:
        raise ValueError("Không có samples để học")
    for row in rows:
        _check_row(row)
    count, dim = len(rows), len(FEATURES)
    means = [sum(row["features"][j] for row in rows)/count for j in range(dim)]
    scales = [max(1e-6, math.sqrt(sum((row["features"][j]-means[j])**2
                                     for row in rows)/count)) for j in range(dim)]
    xs = [[1.0] + [(v-m)/s for v, m, s in zip(row["features"], means, scales)] for row in rows]
    ys = [[v/max(1, b) for v, b in zip(row["future"], row["future_bounds"])] for row in rows]
    n = dim + 1
    gram = [[sum(x[i]*x[j] for x in xs) for j in range(n)] for i in range(n)]
    for i in range(n):
        gram[i][i] += regularization if i else 1e-8
    heads = [_solve(gram, [sum(x[j]*y[h] for x, y in zip(xs, ys)) for j in range(n)])
             for h in range(3)]
    return {"features": list(FEATURES), "means": means, "scales": scales, "heads": heads,
            "regularization": regularization,
            "train_mean": [sum(y[h] for y in ys)/count for h in range(3)],
            "train_min": [min(row["features"][j] for row in rows) for j in range(dim)],
            "train_max": [max(row["features"][j] for row in rows) for j in range(dim)]}


def validate_model(model):
    dim = len(FEATURES)
    if (model.get("features") != list(FEATURES) or len(model.get("means", [])) != dim
            or len(model.get("scales", [])) != dim or len(model.get("heads", [])) != 3
            or any(len(h) != dim+1 for h in model["heads"])):
        raise ValueError("Model feature contract không tương thích")
    values = model["means"] + model["scales"] + [v for head in model["heads"] for v in head]
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            or any(s <= 0 for s in model["scales"])):
        raise ValueError("Model có hệ số không hữu hạn/scale không dương")


def predict_future(model, features, bounds):
    """Return bounded expected gains for the remaining days (not today's gains)."""
    validate_model(model)
    if (len(features) != len(FEATURES) or len(bounds) != 3
            or not all(math.isfinite(v) for v in features+bounds) or any(b < 0 for b in bounds)):
        raise ValueError("Inference features/bounds không hợp lệ")
    x = [1.0] + [(v-m)/s for v, m, s in zip(features, model["means"], model["scales"])]
    raw = [sum(a*b for a, b in zip(head, x)) for head in model["heads"]]
    return [min(1.0, max(0.0, v))*bound for v, bound in zip(raw, bounds)]


def selection_key(model, row):
    future = predict_future(model, row["features"], row["future_bounds"])
    gain = row["gain_today"]
    # Round the integer L/D tiers rather than allowing tiny regression noise
    # in a predicted type to swamp all predicted servings.
    return (gain[0] + round(future[0]), gain[1] + round(future[1]), gain[2] + future[2])


def metrics(model, groups):
    rows = _rows(groups)
    if not rows:
        raise ValueError("Không đánh giá samples rỗng")
    losses = [0.0]*3
    constant_losses = [0.0]*3
    for row in rows:
        _check_row(row)
        pred = predict_future(model, row["features"], row["future_bounds"])
        for h in range(3):
            bound = row["future_bounds"][h]
            losses[h] += ((pred[h]-row["future"][h])/max(1, bound))**2
            constant = model["train_mean"][h]*bound
            constant_losses[h] += ((constant-row["future"][h])/max(1, bound))**2
    wins, ties, failures, informative, correct = 0, 0, 0, 0, 0
    per_group = []
    for group in groups:
        choices = group["rows"]
        baseline = next(row for row in choices if row["baseline"])
        chosen = max(choices, key=lambda row: selection_key(model, row))
        outcome = (tuple(chosen["total_gain"]) > tuple(baseline["total_gain"])) - (
            tuple(chosen["total_gain"]) < tuple(baseline["total_gain"]))
        wins += outcome > 0
        ties += outcome == 0
        failures += outcome < 0
        for i, a in enumerate(choices):
            for b in choices[i+1:]:
                if a["gain_today"] != b["gain_today"] or a["total_gain"] == b["total_gain"]:
                    continue
                informative += 1
                expected = (tuple(a["total_gain"]) > tuple(b["total_gain"]))
                actual_a, actual_b = selection_key(model, a), selection_key(model, b)
                if actual_a != actual_b and ((actual_a > actual_b) == expected):
                    correct += 1
        per_group.append({"group": group["id"], "baseline_gain": baseline["total_gain"],
                          "chosen_gain": chosen["total_gain"], "chosen_candidate": chosen["candidate"]})
    mse = [v/len(rows) for v in losses]
    return {"groups": len(groups), "rows": len(rows), "normalized_mse": mse,
            "mean_mse": sum(mse)/3, "constant_baseline_mse": [v/len(rows) for v in constant_losses],
            "counterfactual_selection": {"wins": wins, "ties": ties, "losses": failures},
            "same_today_score_pairs": informative,
            "same_today_pair_accuracy": correct/informative if informative else None,
            "per_group": per_group,
            "warning": "Counterfactual rollout labels, not wins in deployed full matches"}


def fit(root):
    root = Path(root)
    with writer_lock(root):
        manifest, cfg = verify(root)
        train, train_sha = read_groups(root, "train", cfg)
        validation, validation_sha = read_groups(root, "validation", cfg)
        candidates = []
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            model = fit_ridge(train, alpha)
            candidates.append((metrics(model, validation)["mean_mse"], alpha, model))
        _, _, model = min(candidates, key=lambda row: (row[0], row[1]))
        artifact = {"schema": 1, "kind": "hexudon-future-value-ridge", "experimental": True,
                    "baseline_id": manifest["baseline_id"], "value_id": cfg["value_id"],
                    "automatic_live_deployment": False, "model": model,
                    "training_shards": train_sha, "validation_shards": validation_sha,
                    "regularization_selection": "validation MSE; test never read",
                    "validation_scores": [{"alpha": a, "mean_mse": e} for e, a, _ in candidates]}
        artifact["model_id"] = digest(json_bytes(artifact))
        atomic_json(root / "value-model.json", artifact)
        report = {"model_id": artifact["model_id"], "train": metrics(model, train),
                  "validation": metrics(model, validation)}
        atomic_json(root / "value-fit-report.json", report)
        return report


def evaluate(root, split="test"):
    if split not in ("train", "validation", "test"):
        raise ValueError("split không hợp lệ")
    root = Path(root)
    with writer_lock(root):
        manifest, cfg = verify(root)
        artifact = read_json(root / "value-model.json")
        identity = dict(artifact)
        claimed = identity.pop("model_id")
        if (digest(json_bytes(identity)) != claimed or artifact["kind"] != "hexudon-future-value-ridge"
                or artifact["value_id"] != cfg["value_id"]
                or artifact["baseline_id"] != manifest["baseline_id"]):
            raise ValueError("Value model không thuộc run/dataset này")
        model = artifact["model"]
        validate_model(model)
        groups, checksums = read_groups(root, split, cfg)
        # Even train/validation shards cannot silently change after model fitting.
        for known, key in (("train", "training_shards"), ("validation", "validation_shards")):
            _, actual = read_groups(root, known, cfg)
            if actual != artifact[key]:
                raise ValueError("Samples đã thay đổi sau khi fit")
        report = {"model_id": claimed, "split": split, "sample_shards": checksums,
                  "metrics": metrics(model, groups)}
        atomic_json(root / "value-evaluations" / (split + ".json"), report)
        return report
