"""Reproducible, family-separated scenario corpus and bounded training batches.

Known web maps are development/training data, never an independent test set.
Validation and test families are generated separately before any training.
"""

import argparse
from collections import Counter, deque
import copy
import hashlib
import json
from pathlib import Path
import random

from .. import rules
from ..geometry import hex_distance
from ..model import MatchConfig, Spot
from ..offline.mapgen import generate


PROFILES = ("scarce-stock", "rich-stock", "uneven-stock", "clustered-shops",
            "spread-shops", "few-shops", "many-shops", "rare-brands", "many-brands",
            "fleet-and-starts", "low-fuel", "short-days", "alternating-days",
            "road-costs", "traffic", "mixed")
SHAPES = ((8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (32, 32), (8, 32), (32, 8))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def case_identity(config):
    # Clock changes and different idle opponent counts alone are not new maps.
    return digest({k: v for k, v in config.items() if k not in ("startsAt", "daySeconds", "players")})


def family_identity(cfg):
    return digest({"width": cfg.width, "height": cfg.height, "cells": cfg.cells})


def _reachable(cfg):
    seen, queue = {cfg.agent_starts[0]}, deque([cfg.agent_starts[0]])
    while queue:
        for nxt in cfg.neighbors[queue.popleft()]:
            if nxt >= 0 and nxt not in seen and cfg.cells[nxt] != rules.POND:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def validate_config(cfg):
    """Check generator/model constraints, anchors, stock, and reachability.

    This does not prove that every brand is collectible within the day budget,
    or that an arbitrary planner output is valid; simulator checks still apply.
    """
    if not (8 <= cfg.width <= 32 and 8 <= cfg.height <= 32):
        raise ValueError("Unsupported dimensions")
    if len(cfg.cells) != cfg.width * cfg.height or not set(cfg.cells) <= {0, 1, 2, 3}:
        raise ValueError("Invalid terrain")
    if not (3 <= cfg.n_agents <= 8 and cfg.n_agents <= len(cfg.spots) <= max(cfg.width, cfg.height)):
        raise ValueError("Invalid fleet/shop counts")
    if len(set(cfg.agent_starts)) != cfg.n_agents or len({s.pos for s in cfg.spots}) != len(cfg.spots):
        raise ValueError("Duplicate anchors")
    if not (4 <= cfg.n_days <= 10 and len(cfg.day_seconds) == cfg.n_days):
        raise ValueError("Invalid day count")
    if cfg.fuel_limits < 2 or cfg.players < 4 or not (1 <= cfg.busy_threshold <= cfg.jammed_threshold):
        raise ValueError("Invalid fuel/traffic configuration")
    if any(v <= 0 for v in cfg.day_steps + cfg.day_seconds):
        raise ValueError("Invalid day budget")
    if any(not (0 <= s.brand and 1 <= s.stocks <= cfg.n_agents) for s in cfg.spots):
        raise ValueError("Invalid brand/stock")
    anchors = set(cfg.agent_starts) | {s.pos for s in cfg.spots}
    if any(not (0 <= c < cfg.n_cells) or cfg.cells[c] != rules.PLAIN for c in anchors):
        raise ValueError("Anchors must be on plain cells")
    if not anchors <= _reachable(cfg):
        raise ValueError("Disconnected anchors")


def _positions(cfg, pool, count, rng, spread=False):
    pool = list(pool)
    rng.shuffle(pool)
    if not spread:
        anchor = pool[0]
        return sorted(pool, key=lambda c: hex_distance(anchor, c, cfg.width))[:count]
    selected = [pool.pop()]
    while len(selected) < count:
        nxt = max(pool, key=lambda c: min(hex_distance(c, p, cfg.width) for p in selected))
        selected.append(nxt)
        pool.remove(nxt)
    return selected


def vary(base, profile, seed):
    cfg, rng = copy.deepcopy(base), random.Random(seed)
    reachable = _reachable(cfg)
    plains = [i for i, terrain in enumerate(cfg.cells) if terrain == rules.PLAIN and i in reachable]
    lower, upper = cfg.width + cfg.height, 4 * (cfg.width + cfg.height)
    # Genuine budget variation makes repeated themes distinct, not extra seeds
    # or timestamps attached to exactly the same scenario.
    cfg.day_steps = [max(lower, min(upper, int(round(v * rng.uniform(0.85, 1.15)))))
                     for v in cfg.day_steps]
    if profile in ("fleet-and-starts", "mixed"):
        max_agents = min(8, len(cfg.spots), len(plains) - len(cfg.spots))
        count = rng.randint(3, max_agents)
        cfg.agent_starts = rng.sample([p for p in plains if p not in cfg.spot_index_at], count)
    if profile in ("few-shops", "many-shops", "mixed"):
        count = (rng.randint(len(cfg.agent_starts), max(len(cfg.agent_starts), len(cfg.spots)//2))
                 if profile == "few-shops" else min(max(cfg.width, cfg.height), len(plains)-len(cfg.agent_starts)))
        pool = [p for p in plains if p not in cfg.agent_starts]
        cfg.spots = [Spot(i % max(1, min(len(base.brands), count)), p, 1)
                     for i, p in enumerate(rng.sample(pool, count))]
    if profile in ("clustered-shops", "spread-shops"):
        pool = [p for p in plains if p not in cfg.agent_starts]
        positions = _positions(cfg, pool, len(cfg.spots), rng, spread=profile == "spread-shops")
        cfg.spots = [Spot(s.brand, p, s.stocks) for s, p in zip(cfg.spots, positions)]
    count = len(cfg.spots)
    if profile in ("rare-brands", "many-brands", "mixed"):
        brands = count if profile == "many-brands" else rng.randint(2, count)
        labels = list(range(brands)) + [rng.randrange(1, brands) for _ in range(count-brands)]
        rng.shuffle(labels)
        if profile == "rare-brands":
            remote = max(range(count), key=lambda i: min(hex_distance(cfg.spots[i].pos, p, cfg.width)
                                                         for p in cfg.agent_starts))
            zero = labels.index(0)
            labels[remote], labels[zero] = labels[zero], labels[remote]
        cfg.spots = [Spot(b, s.pos, s.stocks) for b, s in zip(labels, cfg.spots)]
    fleet = len(cfg.agent_starts)
    if profile == "scarce-stock":
        stocks = [1] * count
    elif profile == "rich-stock":
        stocks = [rng.randint(max(1, fleet-1), fleet) for _ in cfg.spots]
    elif profile in ("uneven-stock", "mixed"):
        stocks = [rng.randint(1, fleet) for _ in cfg.spots]
    else:
        stocks = [min(fleet, s.stocks) for s in cfg.spots]
    cfg.spots = [Spot(s.brand, s.pos, stock) for s, stock in zip(cfg.spots, stocks)]
    if profile in ("low-fuel", "mixed"):
        cfg.fuel_limits = max(2, int(min(cfg.day_steps) * rng.uniform(0.35, 1.0)))
    if profile == "short-days":
        cfg.day_steps = [rng.randint(lower, min(upper, lower*2)) for _ in cfg.day_steps]
    if profile in ("alternating-days", "mixed"):
        cfg.day_steps = [rng.randint(lower, lower*2) if i % 2 else rng.randint(lower*3, upper)
                         for i in range(rng.randint(4, 10))]
        cfg.day_seconds = [base.day_seconds[0]] * len(cfg.day_steps)
    if profile in ("road-costs", "mixed"):
        protected = set(cfg.agent_starts) | {s.pos for s in cfg.spots}
        movable = [p for p, terrain in enumerate(cfg.cells) if p not in protected and terrain != rules.POND]
        for p in rng.sample(movable, max(1, len(movable)//10)):
            cfg.cells[p] = rng.choice([rules.PLAIN, rules.ROAD, rules.MOUNTAIN])
    if profile in ("traffic", "mixed"):
        cfg.players = rng.choice([4, 8, 16, 32, 59])
        cfg.busy_threshold = rng.randint(1, 5)
        cfg.jammed_threshold = rng.randint(cfg.busy_threshold, 10)
    cfg.starts_at = 0
    cfg.__post_init__()
    validate_config(cfg)
    return cfg


def build(web_maps, out, random_families=24, per_family=64, seed=20261120):
    if random_families < 3 or per_family < 2:
        raise ValueError("Need >=3 fresh families and >=2 cases per family")
    if Path(out).exists():
        raise ValueError("Output exists; choose a new directory, never overwrite a corpus")
    families, geometry = [], set()
    for path in sorted(Path(web_maps).glob("*.json")):
        if path.name == "manifest.json":
            continue
        cfg = MatchConfig.from_json(json.loads(path.read_text(encoding="utf-8")))
        cfg.players = max(4, cfg.players)
        cfg.starts_at = 0
        validate_config(cfg)
        fid = family_identity(cfg)
        if fid in geometry:
            continue
        geometry.add(fid)
        families.append({"id": fid, "kind": "known-web", "source": path.name,
                         "split": "train", "config": cfg})
    fresh = []
    for index in range(random_families):
        width, height = SHAPES[index % len(SHAPES)]
        cfg = generate(seed=seed+100000+index, width=width, height=height)
        cfg.players = max(4, cfg.players)
        cfg.starts_at = 0
        validate_config(cfg)
        fid = family_identity(cfg)
        if fid in geometry:
            raise ValueError("Duplicate family; choose another seed")
        geometry.add(fid)
        fresh.append({"id": fid, "kind": "fresh-synthetic", "source": f"seed:{seed+100000+index}",
                      "config": cfg})
    random.Random(seed).shuffle(fresh)
    held = max(1, random_families//4)
    for index, family in enumerate(fresh):
        family["split"] = "test" if index < held else "validation" if index < 2*held else "train"
    families += fresh
    root = Path(out)
    (root/"maps").mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "seed": seed, "per_family": per_family, "families": [], "cases": [],
                "note": "Known web families are training only; no server upload or automatic deployment."}
    seen = set()
    for family in sorted(families, key=lambda f: f["id"]):
        base = family["config"]
        manifest["families"].append({k: v for k, v in family.items() if k != "config"})
        for index in range(per_family):
            profile = "base" if index == 0 else PROFILES[(index-1) % len(PROFILES)]
            for attempt in range(100):
                cfg = copy.deepcopy(base) if profile == "base" else vary(
                    base, profile, f"{seed}:{family['id']}:{index}:{attempt}")
                payload = cfg.to_json()
                identity = case_identity(payload)
                if identity not in seen:
                    break
            else:
                raise ValueError("Cannot create another unique case")
            seen.add(identity)
            relative = f"maps/{identity}.json"
            (root/relative).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            manifest["cases"].append({"id": identity, "family": family["id"], "split": family["split"],
                "profile": profile, "path": relative, "seed": seed+index,
                "width": cfg.width, "height": cfg.height, "agents": cfg.n_agents,
                "shops": len(cfg.spots), "brands": len(cfg.brands), "players": cfg.players,
                "fuel_limit": cfg.fuel_limits, "days": cfg.n_days,
                "min_steps": min(cfg.day_steps), "max_steps": max(cfg.day_steps),
                "min_stock": min(s.stocks for s in cfg.spots),
                "max_stock": max(s.stocks for s in cfg.spots),
                "daily_stock": sum(s.stocks for s in cfg.spots)})
    manifest["counts"] = dict(Counter(case["split"] for case in manifest["cases"]))
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def batch(corpus, split, count, index=0, small=False):
    manifest = json.loads((Path(corpus)/"manifest.json").read_text(encoding="utf-8"))
    if split not in ("train", "validation", "test") or count < 1 or index < 0:
        raise ValueError("Invalid batch arguments")
    groups, owners = {}, {}
    for case in manifest["cases"]:
        previous = owners.setdefault(case["family"], case["split"])
        if previous != case["split"]:
            raise ValueError("Family leaked across corpus splits")
        if case["split"] == split and (not small or case["width"]*case["height"] <= 256):
            groups.setdefault(case["family"], []).append(case)
    ordered = []
    for family, cases in sorted(groups.items()):
        random.Random(f"{manifest['seed']}:{family}").shuffle(cases)
    for offset in range(max((len(cases) for cases in groups.values()), default=0)):
        for family in sorted(groups):
            if offset < len(groups[family]):
                ordered.append(groups[family][offset])
    start = index*count
    chosen = ordered[start:start+count]
    if len(chosen) != count:
        raise ValueError("Batch exceeds available cases; no silent recycling")
    for case in chosen:
        payload = json.loads((Path(corpus)/case["path"]).read_text(encoding="utf-8"))
        if case_identity(payload) != case["id"]:
            raise ValueError("Corpus case was modified")
    return chosen


def init_batch(corpus, run_dir, size=8, index=0, small=False, iterations=20, population=2):
    from .trainer import initialize
    selected = {split: batch(corpus, split, size if split == "train" else 4,
                             index=index if split == "train" else 0, small=small if split == "train" else False)
                for split in ("train", "validation", "test")}
    specs = {split: [(str(Path(corpus)/case["path"]), case["seed"]) for case in cases]
             for split, cases in selected.items()}
    result = initialize(run_dir, specs, iterations=iterations, role_iterations=0,
                        population=population, seed=20261120+index, players=None)
    (Path(run_dir)/"corpus-selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("build")
    create.add_argument("--web-maps", required=True)
    create.add_argument("--out", required=True)
    create.add_argument("--random-families", type=int, default=24)
    create.add_argument("--per-family", type=int, default=64)
    create.add_argument("--seed", type=int, default=20261120)
    init = commands.add_parser("init-batch")
    init.add_argument("--corpus", required=True)
    init.add_argument("--run-dir", required=True)
    init.add_argument("--size", type=int, default=8)
    init.add_argument("--index", type=int, default=0)
    init.add_argument("--small", action="store_true")
    init.add_argument("--iterations", type=int, default=20)
    init.add_argument("--population", type=int, default=2)
    args = parser.parse_args()
    if args.action == "build":
        manifest = build(args.web_maps, args.out, args.random_families, args.per_family, args.seed)
        print(json.dumps({"families": len(manifest["families"]), "unique_cases": len(manifest["cases"]),
                          "splits": manifest["counts"]}))
    else:
        result = init_batch(args.corpus, args.run_dir, args.size, args.index, args.small,
                            args.iterations, args.population)
        print("Frozen batch baseline:", result["baseline_id"])


if __name__ == "__main__":
    main()
