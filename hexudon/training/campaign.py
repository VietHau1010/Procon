"""Bounded training campaign against a named, immutable existing model.

Tests are consumed only by finalists; a later finalist gets fresh random maps.
This module never changes a running competition client or the reference run.
"""

import argparse
import json
import math
from pathlib import Path
import time

from . import trainer
from .evaluator import evaluate
from .storage import (atomic_json, digest, json_bytes, load_policy, policy_artifact,
                      read_json, validate_weights, verify_run, writer_lock)


def convincing(report, min_cases=20, win_rate=0.6):
    """A conservative empirical LDQ threshold, not a universal guarantee.

    A two-sided exact sign test ignores ties. Fresh case counts and zero losses
    are required, so repeatedly testing until one lucky match is not enough.
    """
    n = report["wins"] + report["ties"] + report["losses"]
    decisive = report["wins"] + report["losses"]
    sign_p = min(1.0, 2.0 ** (1 - decisive)) if report["losses"] == 0 else 1.0
    return (n >= min_cases and report["passed"] and report["wins"] / n >= win_rate
            and sign_p <= 0.01)


def _case_specs(train_seed=10):
    return {
        "train": [(name, seed) for name in ("random10", "random12", "random16")
                  for seed in (train_seed, train_seed + 1)],
        "validation": [(name, seed) for name in ("random14", "random20")
                       for seed in (1000, 1001)] + [("sample_16", 0), ("sample_24", 0)],
        "test": [(name, seed) for name in ("random16", "random20", "random24", "random8x32")
                 for seed in range(20000, 20006)],
    }


def initialize(directory, reference_run, iterations=150, role_iterations=10, seed=20261120):
    reference_manifest = verify_run(reference_run)
    reference_weights = load_policy(reference_run)
    manifest = trainer.initialize(directory, _case_specs(), iterations=iterations,
                                  role_iterations=role_iterations, population=8, seed=seed)
    if reference_manifest["sources"] != manifest["sources"]:
        raise ValueError("Planner hiện hành khác code của reference model; không chỉ so trọng số")
    validate_weights(reference_weights, manifest["baseline_weights"])
    root = Path(directory)
    with writer_lock(root):
        cfg = {"schema": 1, "baseline_id": manifest["baseline_id"],
               "reference_run": str(Path(reference_run).resolve()),
               "reference_baseline_id": reference_manifest["baseline_id"],
               "reference_model": read_json(Path(reference_run) / "champion.json"),
               "reference_weights": reference_weights,
               "min_test_cases": 20, "min_win_rate": 0.6,
               "campaign_runtime_hash": digest(Path(__file__).read_bytes()),
               "metric": "L,D,Q only; equal ALNS iterations; not live response-time ranking"}
        cfg["campaign_id"] = digest(json_bytes(cfg))
        atomic_json(root / "campaign.json", cfg)
        atomic_json(root / "reference.json", policy_artifact(manifest, reference_weights, "reference"))
        atomic_json(root / "campaign-state.json", {
            "campaign_id": cfg["campaign_id"], "trials": 0, "accepted": 0,
            "weights": reference_weights, "best_trial": "reference", "audits": 0,
            "qualified": False, "compute_seconds": 0.0})
    return cfg


def _fresh_audit(root, manifest, index):
    if index == 0:
        return "test", manifest["datasets"]["test"]
    # This split is generated BEFORE seeing this audit's outcomes. Earlier
    # tests are not re-used for choosing the next model. Store a checksum so
    # an interrupted evaluation can be repeated on exactly the same maps.
    from ..offline.mapgen import load_or_generate
    split = f"campaign-audit-{index:04d}"
    path = root / "data" / (split + ".json")
    cases = []
    old_ids = {c["id"] for key in ("train", "validation", "test")
               for c in read_json(root / "data" / (key + ".json"))}
    for name in ("random16", "random20", "random24", "random8x32"):
        for seed in range(20000 + 100000*index, 20006 + 100000*index):
            c = load_or_generate(name, seed)
            c.players = manifest["settings"]["players"]
            c.starts_at = 0
            obj = c.to_json()
            identity = trainer.map_identity(obj)
            if identity in old_ids:
                raise ValueError("Fresh audit trùng map đã biết")
            old_ids.add(identity)
            cases.append({"id": identity, "name": name, "seed": seed, "config": obj})
    if path.exists():
        if read_json(path) != cases:
            raise ValueError("Dataset audit đã thay đổi")
    else:
        atomic_json(path, cases)
    return split, digest(path.read_bytes())


def run(directory, max_trials=128, hours=2, jobs=4):
    """Stop on convincing fresh audit OR a bounded resource budget.

    Budget is checked between complete evaluations, so an in-flight match can
    finish past the nominal time. No success is claimed for a budget stop.
    """
    if max_trials < 1 or hours <= 0 or not math.isfinite(hours) or jobs < 1:
        raise ValueError("max_trials, hours, jobs phải dương")
    root = Path(directory)
    with writer_lock(root):
        manifest = verify_run(root)
        cfg = read_json(root / "campaign.json")
        identity = dict(cfg)
        claimed = identity.pop("campaign_id")
        if (digest(json_bytes(identity)) != claimed
                or cfg["baseline_id"] != manifest["baseline_id"]
                or cfg["campaign_runtime_hash"] != digest(Path(__file__).read_bytes())):
            raise ValueError("Campaign/reference/runtime đã thay đổi")
        state = read_json(root / "campaign-state.json")
        if state["campaign_id"] != cfg["campaign_id"]:
            raise ValueError("Campaign checkpoint không đúng")
        reference = cfg["reference_weights"]
        validate_weights(state["weights"], manifest["baseline_weights"])
        if state["qualified"]:
            print("Campaign đã đạt cổng kiểm định; xem qualified.json", flush=True)
            return state
        start = time.monotonic()
        last_checkpoint = start
        stop_trial = state["trials"] + max_trials
        try:
            while state["trials"] < stop_trial and time.monotonic() - start < hours*3600:
                index = state["trials"]
                trial = f"trial-{index:05d}"
                weights = trainer.mutate(state["weights"], reference,
                                         manifest["settings"]["seed"], index // 8, index % 8)
                validate_weights(weights, manifest["baseline_weights"])
                report = {"trial": trial, "weights": weights, "comparisons": {},
                          "parent": state["best_trial"], "campaign_id": cfg["campaign_id"]}
                comparisons = report["comparisons"]
                t = comparisons["train_reference"] = evaluate(root, "train", reference, weights, manifest, jobs)
                accepted = t["passed"] and t["wins"] > 0
                if accepted and state["weights"] != reference:
                    t2 = comparisons["train_champion"] = evaluate(root, "train", state["weights"], weights, manifest, jobs)
                    accepted = t2["passed"] and t2["wins"] > 0
                if accepted:
                    v = comparisons["validation_reference"] = evaluate(root, "validation", reference, weights, manifest, jobs)
                    accepted = v["passed"]
                if accepted and state["weights"] != reference:
                    v2 = comparisons["validation_champion"] = evaluate(root, "validation", state["weights"], weights, manifest, jobs)
                    accepted = v2["passed"]
                report["accepted_development"] = accepted
                if accepted:
                    state["weights"] = weights
                    state["best_trial"] = trial
                    state["accepted"] += 1
                    atomic_json(root / "campaign-models" / (trial + ".json"), policy_artifact(manifest, weights, trial))
                    # Do not burn test maps on a model that only beats one small
                    # training example. It first needs broad validation wins.
                    v = comparisons["validation_reference"]
                    if v["wins"] >= math.ceil(cfg["min_win_rate"] * len(v["rows"])):
                        split, checksum = _fresh_audit(root, manifest, state["audits"])
                        atomic_json(root / "audit-plan.json", {
                            "trial": trial, "split": split, "dataset_sha256": checksum})
                        audit = comparisons["fresh_test_reference"] = evaluate(root, split, reference, weights, manifest, jobs)
                        state["audits"] += 1
                        report["audit_dataset_sha256"] = checksum
                        if convincing(audit, cfg["min_test_cases"], cfg["min_win_rate"]):
                            state["qualified"] = True
                            atomic_json(root / "qualified.json", {
                                "model": policy_artifact(manifest, weights, trial),
                                "reference": cfg["reference_model"], "audit": audit,
                                "dataset_sha256": checksum, "metric": cfg["metric"]})
                atomic_json(root / "campaign-trials" / (trial + ".json"), report)
                state["trials"] += 1
                now = time.monotonic()
                state["compute_seconds"] += now - last_checkpoint
                last_checkpoint = now
                atomic_json(root / "campaign-state.json", state)
                atomic_json(root / "campaign-champion.json", policy_artifact(
                    manifest, state["weights"], state["best_trial"]))
                print(f"{trial}: {t['wins']}W/{t['ties']}T/{t['losses']}L train; "
                      f"accepted={accepted}; audits={state['audits']}; qualified={state['qualified']}", flush=True)
                if state["qualified"]:
                    break
        except KeyboardInterrupt:
            print("Đã dừng ở checkpoint trial hoàn tất cuối cùng.", flush=True)
        print(json.dumps({k: v for k, v in state.items() if k != "weights"}), flush=True)
        return state


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--reference-run", help="only needed when initializing a new campaign")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--role-iterations", type=int, default=10)
    p.add_argument("--max-trials", type=int, default=128)
    p.add_argument("--hours", type=float, default=2)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    if not Path(args.run_dir).exists():
        if not args.reference_run:
            p.error("new campaign requires --reference-run")
        initialize(args.run_dir, args.reference_run, args.iterations, args.role_iterations)
    run(args.run_dir, args.max_trials, args.hours, args.jobs)


if __name__ == "__main__":
    main()
