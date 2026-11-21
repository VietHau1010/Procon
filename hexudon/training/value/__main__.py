"""python -m hexudon.training.value {init,collect,fit,evaluate}."""

import argparse
import json

from . import data, model


def main():
    parser = argparse.ArgumentParser(__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "collect", "fit", "evaluate"):
        p = actions.add_parser(name)
        p.add_argument("--run-dir", required=True)
        if name == "init":
            for split, maps, seeds in (("train", "random8,random10", "10,11"),
                                       ("validation", "random12", "1000,1001"),
                                       ("test", "random14", "2000,2001")):
                p.add_argument("--" + split + "-maps", default=maps)
                p.add_argument("--" + split + "-seeds", default=seeds)
            p.add_argument("--iterations", type=int, default=20)
            p.add_argument("--rollout-iterations", type=int, default=10)
            p.add_argument("--candidates", type=int, default=4)
            p.add_argument("--states-per-map", type=int, default=3)
            p.add_argument("--players", type=int, default=4)
            p.add_argument("--reference-run")
        if name in ("collect", "evaluate"):
            p.add_argument("--split", choices=("train", "validation", "test"), default="train" if name == "collect" else "test")
        if name == "collect":
            p.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "init":
            specs = {split: [(n.strip(), int(s.strip()))
                             for n in getattr(args, split + "_maps").split(",") if n.strip()
                             for s in getattr(args, split + "_seeds").split(",") if s.strip()]
                     for split in ("train", "validation", "test")}
            cfg = data.initialize(args.run_dir, specs, args.iterations, args.rollout_iterations,
                                  args.candidates, args.states_per_map, args.players, args.reference_run)
            print("Value run:", cfg["value_id"][:12])
        elif args.command == "collect":
            n = data.collect(args.run_dir, args.split, args.jobs)
            print("Collected new cases:", n)
        elif args.command == "fit":
            report = model.fit(args.run_dir)
            print(json.dumps({split: {k: v for k, v in report[split].items() if k != "per_group"}
                              for split in ("train", "validation")}, indent=2))
        else:
            report = model.evaluate(args.run_dir, args.split)
            print(json.dumps({k: v for k, v in report["metrics"].items() if k != "per_group"}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print("Value training:", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
