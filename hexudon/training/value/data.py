"""Counterfactual candidate states labelled by full remaining-match rollouts."""

from concurrent.futures import ProcessPoolExecutor
import importlib
from pathlib import Path

from .. import trainer
from ..storage import (atomic_json, digest, json_bytes, load_baseline, load_policy,
                       read_json, verify_run, writer_lock)
from .features import FEATURES, extract, future_bounds


def modules_for(root):
    api = load_baseline(root)
    prefix = api["model"].__package__
    for key, suffix in (("engine", ".planner_contest.engine"), ("pathing", ".pathing"),
                        ("traffic", ".traffic"), ("roles", ".planner_contest.roles"),
                        ("rules", ".rules")):
        api[key] = importlib.import_module(prefix + suffix)
    return api


def initialize(root, specs, iterations=20, rollout_iterations=10, candidates=4,
               states_per_map=3, players=4, reference_run=None):
    if min(iterations, rollout_iterations) < 0 or candidates < 2 or states_per_map < 1:
        raise ValueError("iterations >=0, candidates >=2, states_per_map >=1")
    reference = verify_run(reference_run) if reference_run else None
    weights = load_policy(reference_run) if reference_run else None
    manifest = trainer.initialize(root, specs, iterations=iterations, role_iterations=0, players=players)
    if reference and reference["sources"] != manifest["sources"]:
        raise ValueError("Reference planner khác mã snapshot; không chỉ chuyển trọng số")
    obj = {"schema": 1, "kind": "multi-day-value-dataset", "baseline_id": manifest["baseline_id"],
           "weights": weights or manifest["baseline_weights"],
           "settings": {"iterations": iterations, "rollout_iterations": rollout_iterations,
                        "candidates": candidates, "states_per_map": states_per_map, "teams": 4},
           "features": list(FEATURES), "automatic_live_deployment": False,
           "runtime": {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}}
    obj["value_id"] = digest(json_bytes(obj))
    atomic_json(Path(root) / "value-config.json", obj)
    return obj


def verify(root):
    manifest = verify_run(root)
    cfg = read_json(Path(root) / "value-config.json")
    identity = dict(cfg)
    claimed = identity.pop("value_id")
    if (cfg["schema"] != 1 or cfg["features"] != list(FEATURES)
            or digest(json_bytes(identity)) != claimed or cfg["baseline_id"] != manifest["baseline_id"]):
        raise ValueError("Value config không hợp lệ")
    for name, checksum in cfg["runtime"].items():
        if digest((Path(__file__).parent / name).read_bytes()) != checksum:
            raise ValueError(f"Value runtime đã thay đổi: {name}; tạo run mới")
    return manifest, cfg


def _context(api, cfg, state, team, forecast):
    day, status = state["day"], state["status"]
    predicted = forecast.predict(status, state["yesterday_status"],
                                 own_stay_yesterday=state["previous_stay"][team], floor_today=day == 1)
    return api["engine"].make_context(cfg, day, state["agents"][team], status,
                                      [s.stocks for s in cfg.spots], state["visited"][team],
                                      weights=state["weights"], forecast_status=predicted)


def _simulate(api, cfg, state, team, plan):
    result = api["simulator"].run_day(cfg, cfg.steps_of_day(state["day"]), state["agents"][team],
                                       plan, [s.stocks for s in cfg.spots], state["status"], strict_fuel=True)
    if not result.ok:
        raise ValueError(f"Rollout không hợp lệ: day={state['day']} team={team}: {result.error}")
    return result


def _advance(api, cfg, state, results):
    """New independent branch, full-team traffic; stocks reset in each context."""
    combined = {}
    for r in results:
        for cell, count in r.stay.items():
            combined[cell] = combined.get(cell, 0) + count
    history = (state["history"] + [combined])[-2:]
    return {"day": state["day"]+1, "weights": state["weights"], "history": history,
            "status": api["simulator"].road_status_from_history(cfg, history),
            "yesterday_status": dict(state["status"]),
            "previous_stay": [dict(r.stay) for r in results],
            "agents": [[api["model"].AgentState(a.kind, p, f) for a, p, f in
                        zip(old, r.end_pos, r.end_fuel)] for old, r in zip(state["agents"], results)],
            "visited": [set(v) | r.brands for v, r in zip(state["visited"], results)]}


def _results(api, cfg, state, forecasts, iterations, seed):
    results, contexts, plans = [], [], []
    for team in range(4):
        ctx = _context(api, cfg, state, team, forecasts[team])
        if team == 0:
            dec = api["engine"].plan_day(ctx, max_iterations=iterations, seed=seed)
        else:
            _, dec = api["engine"].safe_solution(ctx, seed=seed + 101*team)
        result = _simulate(api, cfg, state, team, dec.plan)
        results.append(result)
        plans.append(dec.plan)
        contexts.append(ctx)
    return results, contexts, plans


def rollout(api, cfg, state, forecasts, iterations, seed):
    """Return FUTURE gains, excluding the candidate's just-completed day."""
    prior = set(state["visited"][0])
    d, q, count = 0, 0, 0
    while state["day"] < cfg.n_days:
        results, _, _ = _results(api, cfg, state, forecasts, iterations, seed)
        d += len(results[0].brands)
        q += results[0].servings
        state = _advance(api, cfg, state, results)
        count += 1
    return [len(state["visited"][0] - prior), d, q], count


def collect_case(job):
    root, case, config = job
    api = modules_for(root)
    cfg = api["model"].MatchConfig.from_json(case["config"])
    weights = api["solution"].Weights(**config["weights"])
    roles = api["roles"].choose_roles_for_match(cfg)
    if roles.count(1) != 1:
        raise ValueError("Teacher phải có đúng một xe tiếp")
    settings = config["settings"]
    forecasts = [api["traffic"].TrafficForecast(cfg, n_patrols=roles.count(0)) for _ in range(4)]
    state = {"day": 0, "status": {}, "yesterday_status": None, "previous_stay": [None]*4,
             "history": [], "weights": weights, "visited": [set() for _ in range(4)],
             "agents": [[api["model"].AgentState(k, p, cfg.fuel_limits)
                         for k, p in zip(roles, cfg.agent_starts)] for _ in range(4)]}
    available = list(range(max(1, cfg.n_days - 1)))
    n = min(settings["states_per_map"], len(available))
    decision_days = {available[round(i*(len(available)-1)/max(1, n-1))] for i in range(n)}
    groups = []
    while state["day"] < cfg.n_days:
        seed = case["seed"]
        results, contexts, plans = _results(api, cfg, state, forecasts, settings["iterations"], seed)
        if state["day"] in decision_days:
            ctx = contexts[0]
            candidates = [plans[0]]
            for index in range(1, settings["candidates"]):
                if index == settings["candidates"] - 1:
                    candidates.append(api["simulator"].idle_plan(cfg, state["agents"][0], ctx.steps, state["status"]))
                else:
                    dec = api["engine"].plan_day(ctx, max_iterations=settings["iterations"],
                                                  seed=seed + 7919*index)
                    candidates.append(dec.plan)
            rows, seen = [], set()
            for index, plan in enumerate(candidates):
                r = _simulate(api, cfg, state, 0, plan)
                fingerprint = digest(json_bytes([r.end_pos, r.end_fuel, sorted(r.brands),
                                                  sorted(r.stay.items()), r.servings]))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                gain = [len(r.brands - state["visited"][0]), len(r.brands), r.servings]
                # Opponents' plans today are held fixed across branches. Future
                # routes respond to each branch's altered positions and traffic.
                branch = _advance(api, cfg, state, [r] + results[1:])
                future, done = rollout(api, cfg, branch, forecasts, settings["rollout_iterations"], seed)
                if done != ctx.days_left:
                    raise ValueError("Không dùng nhãn từ rollout chưa đủ ngày")
                bounds = future_bounds(ctx, r)
                if any(v < 0 or v > bound for v, bound in zip(future, bounds)):
                    raise ValueError("Nhãn rollout vượt cận luật")
                rows.append({"candidate": index, "baseline": index == 0,
                             "features": extract(ctx, r, api, forecasts[0],
                                                 state["yesterday_status"], state["previous_stay"][0]),
                             "gain_today": gain, "future": future, "future_bounds": bounds,
                             "total_gain": [a+b for a, b in zip(gain, future)],
                             "terminal_heuristic": api["objective"].end_value(ctx, r),
                             "plan_sha256": digest(json_bytes(plan)), "rollout_days": done,
                             "end_pos": r.end_pos, "end_fuel": r.end_fuel})
            groups.append({"id": case["id"] + f":day{state['day']}", "day": state["day"],
                           "days_left": ctx.days_left, "rows": rows})
        state = _advance(api, cfg, state, results)
    return {"case": case["id"], "value_id": config["value_id"], "groups": groups}


def read_shard(path, case, config):
    obj = read_json(path)
    checksum = obj.pop("sha256", None)
    if (digest(json_bytes(obj)) != checksum or obj["case"] != case["id"]
            or obj["value_id"] != config["value_id"]):
        raise ValueError(f"Sample shard bị thay đổi: {path}")
    return obj


def collect(root, split="train", jobs=1):
    if split not in ("train", "validation", "test") or jobs < 1:
        raise ValueError("split/jobs không hợp lệ")
    root = Path(root)
    with writer_lock(root):
        _, cfg = verify(root)
        cases = read_json(root / "data" / (split + ".json"))
        work = []
        for case in cases:
            path = root / "samples" / split / (case["id"] + ".json")
            if path.exists():
                read_shard(path, case, cfg)
            else:
                work.append((str(root.resolve()), case, cfg))
        def save(obj):
            obj["sha256"] = digest(json_bytes(obj))
            atomic_json(root / "samples" / split / (obj["case"] + ".json"), obj)
            print(f"{split}: {obj['case'][:10]}: {len(obj['groups'])} states, "
                  f"{sum(len(g['rows']) for g in obj['groups'])} branches", flush=True)
        if work and jobs > 1:
            with ProcessPoolExecutor(max_workers=min(jobs, len(work))) as pool:
                for obj in pool.map(collect_case, work):
                    save(obj)
        else:
            for job in work:
                save(collect_case(job))
        return len(work)


def read_groups(root, split, cfg):
    groups, checksums = [], {}
    for case in read_json(Path(root) / "data" / (split + ".json")):
        path = Path(root) / "samples" / split / (case["id"] + ".json")
        obj = read_shard(path, case, cfg)
        groups.extend(obj["groups"])
        checksums[case["id"]] = digest(path.read_bytes())
    return groups, checksums
