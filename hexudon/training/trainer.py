"""Resumable (1+lambda) evolutionary learning with strict empirical promotion gates."""

from dataclasses import asdict
import math
from pathlib import Path
import random

from ..offline.mapgen import load_or_generate
from .evaluator import evaluate
from .storage import (SCHEMA, SPACE, atomic_json, digest, json_bytes, load_baseline,
                      load_policy, policy_artifact, read_json, snapshot_sources,
                      validate_weights, verify_run, writer_lock)


def map_identity(config):
    # Ignore wall-clock fields/team count: rescheduling the same map is not an
    # independent validation example. Include day budgets, terrain and stocks.
    obj = dict(config)
    for key in ("startsAt", "daySeconds", "players"):
        obj.pop(key, None)
    return digest(json_bytes(obj))


def initialize(directory, specs, iterations=100, role_iterations=10, population=4, seed=0, players=4):
    """specs maps each of train/validation/test to [(map name/path, seed), ...]."""
    if iterations < 0 or role_iterations < 0 or population < 1 or (players is not None and players < 4):
        raise ValueError("iterations >= 0, population >= 1, players >= 4")
    if set(specs) != {"train", "validation", "test"}:
        raise ValueError("Cần đủ train, validation và test")
    datasets, seen = {}, {}
    for split, entries in specs.items():
        cases = []
        for name, case_seed in entries:
            cfg = load_or_generate(str(name), seed=case_seed)
            if cfg.n_agents < 2 or not cfg.n_days:
                raise ValueError("Map huấn luyện phải có >=2 xe và ít nhất một ngày")
            if players is not None:
                cfg.players = players
            elif cfg.players < 4:
                raise ValueError("Per-map players must be >=4 for this four-team evaluator")
            cfg.starts_at = 0
            obj = cfg.to_json()
            identity = map_identity(obj)
            if identity in seen:
                if seen[identity] != split:
                    raise ValueError(f"Rò rỉ dữ liệu: cùng map nằm ở {seen[identity]} và {split}: {name}")
                continue  # Same sample map with several seeds is not extra data.
            seen[identity] = split
            cases.append({"id": identity, "name": str(name), "seed": case_seed, "config": obj})
        if not cases:
            raise ValueError(f"Dataset rỗng: {split}")
        datasets[split] = cases
    root = Path(directory)
    if root.exists():
        raise ValueError("Run đã tồn tại; dùng train run để tiếp tục, hoặc chọn --run-dir mới")
    root.mkdir(parents=True)
    with writer_lock(root):
        sources = snapshot_sources(root)
        api = load_baseline(root)
        baseline = asdict(api["objective"].default_contest_weights())
        validate_weights(baseline, baseline)
        checksums = {}
        for split, cases in datasets.items():
            path = root / "data" / (split + ".json")
            atomic_json(path, cases)
            checksums[split] = digest(path.read_bytes())
        manifest = {"schema": SCHEMA, "algorithm": "elitist-evolution-v1",
                    "sources": sources, "datasets": checksums, "baseline_weights": baseline,
                    "settings": {"iterations": iterations, "role_iterations": role_iterations,
                                 "population": population, "seed": seed, "players": players},
                    "search_space": {k: list(v) for k, v in SPACE.items()},
                    "runtime": {name: digest((Path(__file__).parent / name).read_bytes())
                                for name in ("trainer.py", "evaluator.py", "storage.py")},
                    "promotion": "zero LDQ losses and zero invalid plans",
                    "live_deployment": False}
        manifest["baseline_id"] = digest(json_bytes(manifest))
        atomic_json(root / "manifest.json", manifest)
        atomic_json(root / "baseline.json", policy_artifact(manifest, baseline, "baseline"))
        state = {"schema": SCHEMA, "baseline_id": manifest["baseline_id"],
                 "generation": 0, "next_candidate": 0, "trials": 0,
                 "promotions": 0, "weights": baseline, "champion_trial": "baseline"}
        atomic_json(root / "state.json", state)
        atomic_json(root / "champion.json", policy_artifact(manifest, baseline, "baseline"))
    return manifest


def mutate(parent, baseline, seed, generation, index):
    # Each trial has its own RNG. Resuming after Ctrl-C never relies on how many
    # random calls prior evaluations happened to make.
    rng = random.Random(f"{seed}:{generation}:{index}:evolution-v1")
    child = dict(baseline if index % 4 == 3 else parent)
    names = list(SPACE)
    changed = rng.sample(names, rng.randint(1, min(3, len(names))))
    for name in changed:
        low, high = SPACE[name]
        child[name] = round(min(high, max(low, child[name] * math.exp(rng.gauss(0, 0.45)))), 8)
    return child


def status(directory):
    manifest = verify_run(directory)
    state = read_json(Path(directory) / "state.json")
    if state.get("schema") != SCHEMA or state.get("baseline_id") != manifest["baseline_id"]:
        raise ValueError("Checkpoint không thuộc baseline này")
    validate_weights(state["weights"], manifest["baseline_weights"])
    return manifest, state


def run(directory, generations=1, jobs=1, log=print):
    """Run N additional generations, or zero for continuous learning until Ctrl-C.

    Validation is a reused acceptance set, not an unbiased final test. The test
    split is never consulted or used for promotion in this loop.
    """
    if generations < 0 or jobs < 1:
        raise ValueError("generations >= 0, jobs >= 1")
    root = Path(directory)
    with writer_lock(root):
        manifest, state = status(root)
        base = manifest["baseline_weights"]
        population = manifest["settings"]["population"]
        target = state["generation"] + generations if generations else None
        # state.json is authoritative if the last run was interrupted between
        # committing state and writing its human-readable model export.
        atomic_json(root / "champion.json", policy_artifact(manifest, state["weights"], state["champion_trial"]))
        try:
            while target is None or state["generation"] < target:
                gen, index = state["generation"], state["next_candidate"]
                trial = f"g{gen:05d}-c{index:03d}"
                candidate = mutate(state["weights"], base, manifest["settings"]["seed"], gen, index)
                validate_weights(candidate, base)
                report = {"trial": trial, "weights": candidate, "baseline_id": manifest["baseline_id"],
                          "parent_trial": state["champion_trial"], "comparisons": {}}
                comparisons = report["comparisons"]
                comparisons["train_vs_baseline"] = evaluate(root, "train", base, candidate, manifest, jobs)
                accepted = comparisons["train_vs_baseline"]["passed"]
                # Strict gain must be on training; validation is only a gate.
                accepted = accepted and comparisons["train_vs_baseline"]["wins"] > 0
                if accepted and state["weights"] != base:
                    comparisons["train_vs_champion"] = evaluate(
                        root, "train", state["weights"], candidate, manifest, jobs)
                    accepted = comparisons["train_vs_champion"]["passed"] and comparisons["train_vs_champion"]["wins"] > 0
                if accepted:
                    comparisons["validation_vs_baseline"] = evaluate(root, "validation", base, candidate, manifest, jobs)
                    accepted = comparisons["validation_vs_baseline"]["passed"]
                if accepted and state["weights"] != base:
                    comparisons["validation_vs_champion"] = evaluate(
                        root, "validation", state["weights"], candidate, manifest, jobs)
                    accepted = comparisons["validation_vs_champion"]["passed"]
                report["accepted"] = bool(accepted)
                atomic_json(root / "trials" / (trial + ".json"), report)
                if accepted:
                    state["weights"] = candidate
                    state["champion_trial"] = trial
                    state["promotions"] += 1
                    atomic_json(root / "models" / (trial + ".json"), policy_artifact(manifest, candidate, trial))
                state["trials"] += 1
                state["next_candidate"] += 1
                if state["next_candidate"] == population:
                    state["generation"] += 1
                    state["next_candidate"] = 0
                # Commit one complete, validated trial; interrupted evaluations
                # cannot promote anything and are repeated on resume.
                atomic_json(root / "state.json", state)
                atomic_json(root / "champion.json", policy_artifact(
                    manifest, state["weights"], state["champion_trial"]))
                result = comparisons["train_vs_baseline"]
                log(f"{trial}: train {result['wins']}W/{result['ties']}T/{result['losses']}L; "
                    f"{'ACCEPTED' if accepted else 'rejected'}")
        except KeyboardInterrupt:
            log("Đã dừng. Checkpoint giữ trial hoàn tất cuối cùng; chạy lại train run để tiếp tục.")
        return state


def evaluate_policy(directory, split="test", model_path=None, jobs=1):
    if split not in ("train", "validation", "test") or jobs < 1:
        raise ValueError("split/jobs không hợp lệ")
    root = Path(directory)
    with writer_lock(root):
        manifest = verify_run(root)
        weights = load_policy(root, model_path)
        report = evaluate(root, split, manifest["baseline_weights"], weights, manifest, jobs)
        report["baseline_id"] = manifest["baseline_id"]
        report["weights"] = weights
        report["split"] = split
        # Diagnostic only: this function never changes state or champion.
        atomic_json(root / "evaluations" / (split + "-" + digest(json_bytes(weights))[:12] + ".json"), report)
        return report
