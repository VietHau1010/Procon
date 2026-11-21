"""Build diverse offline cases from maps captured from the web server.

The generated cases are for training and stress testing only. They never claim
to be BTC maps and are never uploaded to the contest server.
"""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..model import MatchConfig
from ..rules import PLAIN, ROAD


def configs_from_replays(paths: Iterable[Path]) -> Dict[str, dict]:
    """Extract unique /config payloads from JSONL web replays."""
    found: Dict[str, dict] = {}
    for path in paths:
        try:
            rows = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in rows:
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if row.get("path") != "/api/game/config" or row.get("status") != 200:
                continue
            payload = row.get("response")
            if not isinstance(payload, dict) or not payload.get("map"):
                continue
            game_id = str((row.get("params") or {}).get("game_id") or path.stem)
            found.setdefault(game_id, payload)
    return found


def _config_payload(payload: dict) -> dict:
    """Return a detached, normalised setup payload suitable for MatchConfig."""
    cfg = MatchConfig.from_json(copy.deepcopy(payload))
    return cfg.to_json()


def _critical(cfg: MatchConfig) -> set:
    return set(cfg.agent_starts) | {spot.pos for spot in cfg.spots}


def _swap_terrain(cfg: MatchConfig, rng: random.Random, swaps: int) -> None:
    """Change road/plain corridors while preserving passability and anchors."""
    critical = _critical(cfg)
    roads = [c for c in cfg.road_cells if c not in critical]
    plains = [c for c, terrain in enumerate(cfg.cells)
              if terrain == PLAIN and c not in critical]
    rng.shuffle(roads)
    rng.shuffle(plains)
    for road, plain in zip(roads[:swaps], plains[:swaps]):
        cfg.cells[road], cfg.cells[plain] = PLAIN, ROAD


def _payload(cfg: MatchConfig) -> dict:
    return cfg.to_json()


def generate_variants(payload: dict, seed: int, count: int = 16) -> List[Tuple[str, dict]]:
    """Generate map-family variants with explicit, reproducible labels.

    Variants preserve the server map's geometry and anchors, while covering
    scarce/abundant stocks, brand rarity, traffic thresholds, day lengths and
    road/plain placement. This creates useful training diversity without
    inventing a different rule set.
    """
    base = MatchConfig.from_json(_config_payload(payload))
    out: List[Tuple[str, dict]] = [("base", _payload(base))]
    rng = random.Random(seed)
    patrols = sum(1 for _ in base.agent_starts)

    for index in range(max(0, count - 1)):
        cfg = copy.deepcopy(base)
        mode = index % 6
        if mode == 0:
            label = "scarce-stock"
            cfg.spots = [type(s)(s.brand, s.pos, 1) for s in cfg.spots]
        elif mode == 1:
            label = "abundant-stock"
            cfg.spots = [type(s)(s.brand, s.pos, max(s.stocks, patrols)) for s in cfg.spots]
        elif mode == 2:
            label = "brand-rarity"
            brands = [s.brand for s in cfg.spots]
            rng.shuffle(brands)
            cfg.spots = [type(s)(brand, s.pos, s.stocks)
                         for s, brand in zip(cfg.spots, brands)]
        elif mode == 3:
            label = "road-layout"
            _swap_terrain(cfg, rng, max(1, len(cfg.road_cells) // 12))
        elif mode == 4:
            label = "short-days"
            cfg.day_steps = [max(1, int(round(v * 0.8))) for v in cfg.day_steps]
        else:
            label = "traffic-thresholds"
            cfg.players = max(1, rng.choice([1, base.players, 4, 10, 59]))
            cfg.busy_threshold = max(1, base.busy_threshold + rng.choice([-1, 0, 1]))
            cfg.jammed_threshold = max(cfg.busy_threshold, base.jammed_threshold + rng.choice([-1, 0, 1]))
        cfg.__post_init__()
        out.append((f"{label}-{index:02d}", _payload(cfg)))
    return out


def write_variants(replay_dir: str, output_dir: str, count: int = 16, seed: int = 0) -> dict:
    source_paths = sorted(Path(replay_dir).glob("*.jsonl"))
    sources = configs_from_replays(source_paths)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": [], "variants": []}
    for source_index, (game_id, payload) in enumerate(sorted(sources.items())):
        source_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        manifest["sources"].append({"game_id": game_id, "sha256": source_hash})
        for label, variant in generate_variants(payload, seed + source_index, count):
            name = f"{game_id[:8]}-{label}.json"
            path = out_dir / name
            path.write_text(json.dumps(variant, indent=2), encoding="utf-8")
            manifest["variants"].append({"source": game_id, "label": label,
                                         "path": str(Path(name))})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", default="replays")
    parser.add_argument("--out", default="artifacts/web-map-variants")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = write_variants(args.replays, args.out, args.count, args.seed)
    print(f"captured {len(manifest['sources'])} maps, wrote {len(manifest['variants'])} variants")


if __name__ == "__main__":
    main()
