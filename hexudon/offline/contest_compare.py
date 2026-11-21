"""Paired matches against a frozen contest package, including daily upper bounds.

python -m hexudon.offline.contest_compare --baseline artifacts/contest-baseline
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from ..planner_contest.bounds import day_bounds
from ..planner_contest.engine import make_context
from ..planner_contest.strategy import MatchStrategist
from .bots import GreedyBot, SampleBot
from .mapgen import load_or_generate
from .runner import run_match


def load_baseline(path, name="hexudon.frozen_contest"):
    if (Path(path) / "baseline" / "planner_contest" / "__init__.py").is_file():
        # Complete snapshots isolate the shared decoder too. A contest-only
        # copy still imports the working tree's ../planner dependencies.
        from ..training.storage import load_baseline as load_snapshot
        return load_snapshot(path)["strategy"].MatchStrategist
    spec = importlib.util.spec_from_file_location(
        name, Path(path) / "__init__.py", submodule_search_locations=[str(Path(path).resolve())])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.MatchStrategist


def source_hashes(path):
    root = Path(path)
    if (root / "baseline" / "planner_contest" / "__init__.py").is_file():
        root = root / "baseline"
        files = root.rglob("*.py")
    else:
        files = root.glob("*.py")
    return {f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(files)}


class ContestBot:
    def __init__(self, cls, cfg, name, seed, iterations, role_iterations, seconds):
        self.name = name
        config_class = getattr(sys.modules.get(cls.__module__), "MatchConfig", type(cfg))
        bot_cfg = cfg if isinstance(cfg, config_class) else config_class.from_json(cfg.to_json())
        self.strategist = cls(bot_cfg, seed=seed, search_iterations=iterations,
                              role_iterations=role_iterations, search_seconds=seconds,
                              role_seconds=3, net_margin=0)
        self.last_response_seconds = 0.0
        self.role_seconds = 0.0
        self.daily_bounds = []

    def select_roles(self):
        began = time.monotonic()
        kinds = self.strategist.select_roles()
        self.role_seconds = time.monotonic() - began
        return kinds

    def plan(self, state):
        cfg = self.strategist.config
        ctx = make_context(cfg, state.day, state.agents, state.road_status,
                           [s.stocks for s in cfg.spots], self.strategist.visited_brands)
        self.daily_bounds.append(day_bounds(ctx))
        began = time.monotonic()

        def submit(plan):
            self.last_response_seconds = time.monotonic() - began
            return True

        return self.strategist.plan_day(state, submit=submit).plan


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", help="Frozen candidate package; omit for working tree")
    p.add_argument("--maps", default="sample_16,sample_24,sample_32,random16,random24")
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--players", type=int, default=4,
                   help="Traffic denominator/team count; use replay's actual count for map-specific tests")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--role-iterations", type=int, default=40)
    p.add_argument("--seconds", type=float, default=None,
                   help="Equal wall-clock daily budget instead of iterations")
    p.add_argument("--fail-on-loss", action="store_true",
                    help="exit nonzero on any paired L/D/Q loss; response time is reported separately")
    p.add_argument("--out", default="artifacts/contest-comparison.json")
    args = p.parse_args()
    baseline = load_baseline(args.baseline)
    candidate = (load_baseline(args.candidate, "hexudon.frozen_candidate")
                 if args.candidate else MatchStrategist)
    rows = []
    for name in args.maps.split(","):
        for seed in range(args.seed, args.seed + args.count):
            cfg = load_or_generate(name, seed)
            cfg.players = args.players
            a = ContestBot(baseline, cfg, "baseline", seed,
                           args.iterations if args.seconds is None else None,
                           args.role_iterations if args.seconds is None else None, args.seconds)
            b = ContestBot(candidate, cfg, "candidate", seed,
                           args.iterations if args.seconds is None else None,
                           args.role_iterations if args.seconds is None else None, args.seconds)
            result = run_match(cfg, [a, b, GreedyBot(cfg, seed=seed), SampleBot(cfg, seed=seed)])
            ta, tb = result.teams[:2]
            row = dict(map=name, seed=seed, baseline=ta.score(), candidate=tb.score(),
                       roles_a=ta.kinds, roles_b=tb.kinds,
                       invalid=[ta.invalid_days, tb.invalid_days],
                       role_seconds=[a.role_seconds, b.role_seconds],
                       plan_seconds=[ta.plan_seconds, tb.plan_seconds],
                       response_seconds=[ta.response_seconds, tb.response_seconds],
                       days=result.days,
                       daily_bounds=[a.daily_bounds, b.daily_bounds],
                       type_ceiling=len(cfg.brands),
                       stock_ceiling=[sum(min(s.stocks, kinds.count(0)) for s in cfg.spots)
                                      for kinds in (ta.kinds, tb.kinds)])
            rows.append(row)
            print(name, seed, ta.score(), tb.score(), flush=True)
    wins = sum(r["candidate"] > r["baseline"] for r in rows)
    ties = sum(r["candidate"] == r["baseline"] for r in rows)
    report = dict(wins=wins, ties=ties, losses=len(rows)-wins-ties,
                  totals_a=[sum(r["baseline"][i] for r in rows) for i in range(3)],
                  totals_b=[sum(r["candidate"][i] for r in rows) for i in range(3)],
                  settings=vars(args),
                   baseline_hashes=source_hashes(args.baseline), rows=rows)
    candidate_path = Path(args.candidate) if args.candidate else Path(__file__).parents[1] / "planner_contest"
    report["candidate_hashes"] = source_hashes(candidate_path)
    report["full_type_days"] = [sum(day["teams"][i]["types"] == r["type_ceiling"]
                                    for r in rows for day in r["days"]) for i in (0, 1)]
    report["stock_ceiling_days"] = [sum(day["teams"][i]["servings"] == r["stock_ceiling"][i]
                                        for r in rows for day in r["days"]) for i in (0, 1)]
    report["days_per_variant"] = sum(len(r["days"]) for r in rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items()
           if k not in ("rows", "settings", "baseline_hashes", "candidate_hashes")})
    if args.fail_on_loss and report["losses"]:
        raise SystemExit(f"non-regression gate failed: {report['losses']} paired L/D/Q losses")


if __name__ == "__main__":
    main()
