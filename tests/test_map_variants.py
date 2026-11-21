"""Web-captured map variants remain valid and reproducible."""

import json
import tempfile
import unittest
from pathlib import Path

from hexudon.model import MatchConfig
from hexudon.offline.map_variants import configs_from_replays, generate_variants, write_variants


class MapVariantsTest(unittest.TestCase):
    def test_extracts_unique_configs_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.jsonl"
            payload = {"startsAt": 0, "daySeconds": [5], "daySteps": [10],
                       "map": {"width": 8, "height": 8, "cells": [[0] * 8 for _ in range(8)]},
                       "spots": [{"brand": 0, "pos": 9, "stocks": 1}],
                       "agents": [0], "fuelLimits": 20, "players": 4,
                       "busyThreshold": 1, "jammedThreshold": 2}
            path.write_text(json.dumps({"path": "/api/game/config", "status": 200,
                                        "params": {"game_id": "web-1"}, "response": payload}) + "\n")
            self.assertEqual(list(configs_from_replays([path])), ["web-1"])

    def test_variants_preserve_anchors_and_are_valid_configs(self):
        from hexudon.offline.mapgen import generate
        base = generate(seed=3, size=8, agents=3, spots=8, chains=4, days=4).to_json()
        variants = generate_variants(base, seed=9, count=13)
        self.assertEqual(len(variants), 13)
        original = MatchConfig.from_json(base)
        for _, payload in variants:
            cfg = MatchConfig.from_json(payload)
            self.assertEqual((cfg.width, cfg.height, len(cfg.spots), cfg.n_agents),
                             (original.width, original.height, len(original.spots), original.n_agents))
            self.assertEqual(cfg.agent_starts, original.agent_starts)
            self.assertEqual([s.pos for s in cfg.spots], [s.pos for s in original.spots])
            self.assertTrue(all(0 <= s.brand for s in cfg.spots))
            self.assertTrue(all(s.stocks >= 1 for s in cfg.spots))

    def test_write_manifest_is_reproducible(self):
        from hexudon.offline.mapgen import generate
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replays"
            out_a = Path(tmp) / "a"
            out_b = Path(tmp) / "b"
            replay_dir.mkdir()
            payload = generate(seed=4, size=8, agents=3, spots=8, days=3).to_json()
            (replay_dir / "x.jsonl").write_text(json.dumps({"path": "/api/game/config",
                "status": 200, "params": {"game_id": "x"}, "response": payload}) + "\n")
            write_variants(str(replay_dir), str(out_a), count=7, seed=2)
            write_variants(str(replay_dir), str(out_b), count=7, seed=2)
            for path in out_a.glob("*.json"):
                self.assertEqual(path.read_bytes(), (out_b / path.name).read_bytes())
