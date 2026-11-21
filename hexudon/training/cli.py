"""CLI plumbing, deliberately separate from all online transports."""

import json

from .trainer import evaluate_policy, initialize, run, status


def _seeds(text):
    try:
        result = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Seeds phải là các số nguyên, ví dụ 0,1,2") from exc
    if not result:
        raise ValueError("Danh sách seed rỗng")
    return result


def command(args):
    try:
        if args.train_action == "init":
            specs = {split: [(name.strip(), seed)
                             for name in getattr(args, split + "_maps").split(",") if name.strip()
                             for seed in _seeds(getattr(args, split + "_seeds"))]
                     for split in ("train", "validation", "test")}
            manifest = initialize(args.run_dir, specs, args.iterations, args.role_iterations,
                                  args.population, args.seed, args.players)
            print(f"Đã đóng băng baseline {manifest['baseline_id'][:12]} tại {args.run_dir}")
        elif args.train_action == "run":
            state = run(args.run_dir, args.generations, args.jobs)
            print(f"generation={state['generation']}, trials={state['trials']}, "
                  f"promotions={state['promotions']}, champion={state['champion_trial']}")
        elif args.train_action == "status":
            manifest, state = status(args.run_dir)
            print(json.dumps({"baseline_id": manifest["baseline_id"], **state}, indent=2))
        else:
            report = evaluate_policy(args.run_dir, args.split, args.model, args.jobs)
            print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Training: {exc}")
        return 2


def add_parser(sub):
    parser = sub.add_parser("train", help="học trọng số chiến thuật offline, không dùng token")
    actions = parser.add_subparsers(dest="train_action", required=True)
    for name in ("init", "run", "status", "evaluate"):
        p = actions.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.set_defaults(func=command)
        if name == "init":
            p.add_argument("--train-maps", default="sample_16,random16")
            p.add_argument("--validation-maps", default="random20,random24")
            p.add_argument("--test-maps", default="sample_24,random28")
            p.add_argument("--train-seeds", default="0,1")
            p.add_argument("--validation-seeds", default="1000,1001")
            p.add_argument("--test-seeds", default="2000,2001")
            p.add_argument("--iterations", type=int, default=100)
            p.add_argument("--role-iterations", type=int, default=10)
            p.add_argument("--population", type=int, default=4)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--players", type=int, default=4)
        if name in ("run", "evaluate"):
            p.add_argument("--jobs", type=int, default=1)
        if name == "run":
            p.add_argument("--generations", type=int, default=1,
                           help="số thế hệ bổ sung; 0 = liên tục tới Ctrl-C")
        if name == "evaluate":
            p.add_argument("--split", choices=["train", "validation", "test"], default="test")
            p.add_argument("--model", help="JSON model; mặc định champion.json của run")
