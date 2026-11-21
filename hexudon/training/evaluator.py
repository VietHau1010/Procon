"""Deterministic paired full matches using only the frozen simulator/planner."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from .storage import load_baseline, read_json


def evaluate_case(job):
    directory, case, left_weights, right_weights, settings = job
    api = load_baseline(directory)
    model, sim = api["model"], api["simulator"]
    cfg = model.MatchConfig.from_json(case["config"])
    cls = api["strategy"].MatchStrategist
    # All teams start a day from the same snapshot and share traffic only.
    bots = [cls(cfg, weights=api["solution"].Weights(**weights), seed=case["seed"],
                 search_iterations=iterations, role_iterations=role_iterations,
                 net_margin=0, log=lambda msg: None,
                 enable_dense=False, fast_roles=True)
            for weights, iterations, role_iterations in (
                (left_weights, settings["iterations"], settings["role_iterations"]),
                (right_weights, settings["iterations"], settings["role_iterations"]),
                (settings["baseline_weights"], 0, 0),
                (settings["baseline_weights"], 0, 0))]
    agents, invalid, kinds, errors = [], [0]*4, [], []
    for i, bot in enumerate(bots):
        try:
            roles = list(bot.select_roles())
            if len(roles) != cfg.n_agents or roles.count(1) != 1:
                raise ValueError("role must contain exactly one tanker")
        except Exception as exc:
            invalid[i] += 1
            errors.append({"team": i, "phase": "roles", "error": repr(exc)})
            roles = [0]*(cfg.n_agents - 1) + [1]
        kinds.append(roles)
        bot.kinds = list(roles)
        agents.append([model.AgentState(k, p, cfg.fuel_limits)
                       for k, p in zip(roles, cfg.agent_starts)])
    visited, daily, servings = [set() for _ in bots], [0]*4, [0]*4
    response_seconds = [0.0]*4
    history, days = [], []
    for day in range(cfg.n_days):
        status = sim.road_status_from_history(cfg, history) if day else {}
        snapshot = [[model.AgentState(a.kind, a.pos, a.fuel) for a in team] for team in agents]
        combined, records = {}, []
        for i, bot in enumerate(bots):
            state = model.DayState(day, snapshot[i], dict(status),
                                   others=[t for j, t in enumerate(snapshot) if j != i])
            began = time.monotonic()
            accepted = [None]

            def submit(plan):
                check = sim.run_day(cfg, cfg.steps_of_day(day), snapshot[i], plan,
                                    [s.stocks for s in cfg.spots], status, strict_fuel=True)
                if not check.ok:
                    invalid[i] += 1
                    return False
                accepted[0] = plan
                return True

            try:
                bot.plan_day(state, submit=submit)
                plan = accepted[0]
                if plan is None:
                    raise ValueError("no valid submission")
            except Exception as exc:
                invalid[i] += 1
                errors.append({"team": i, "phase": "day", "day": day, "error": repr(exc)})
                plan = accepted[0] or sim.wait_plan(cfg.n_agents, cfg.steps_of_day(day))
            result = sim.run_day(cfg, cfg.steps_of_day(day), snapshot[i], plan,
                                 [s.stocks for s in cfg.spots], status, strict_fuel=True)
            response_seconds[i] += time.monotonic() - began
            if not result.ok:
                invalid[i] += 1
                result = sim.run_day(cfg, cfg.steps_of_day(day), snapshot[i],
                                     sim.wait_plan(cfg.n_agents, cfg.steps_of_day(day)),
                                     [s.stocks for s in cfg.spots], status)
            visited[i] |= result.brands
            daily[i] += len(result.brands)
            servings[i] += result.servings
            agents[i] = [model.AgentState(k, p, f) for k, p, f in
                         zip(kinds[i], result.end_pos, result.end_fuel)]
            for cell, count in result.stay.items():
                combined[cell] = combined.get(cell, 0) + count
            records.append({"types": len(result.brands), "servings": result.servings})
        history.append(combined)
        days.append({"day": day, "teams": records[:2]})
    return {"case": case["id"], "map": case["name"], "seed": case["seed"],
            "baseline": [len(visited[0]), daily[0], servings[0]],
            "candidate": [len(visited[1]), daily[1], servings[1]],
            "invalid": invalid, "roles": kinds[:2], "days": days,
            "compute_seconds": response_seconds[:2], "errors": errors}


def summarize(rows):
    if not rows:
        raise ValueError("Không đánh giá dataset rỗng")
    wins = sum(tuple(r["candidate"]) > tuple(r["baseline"]) for r in rows)
    losses = sum(tuple(r["candidate"]) < tuple(r["baseline"]) for r in rows)
    invalid = sum(sum(r["invalid"]) for r in rows)
    return {"wins": wins, "ties": len(rows)-wins-losses, "losses": losses,
            "invalid": invalid, "passed": losses == 0 and invalid == 0,
            "baseline_total": [sum(r["baseline"][i] for r in rows) for i in range(3)],
            "candidate_total": [sum(r["candidate"][i] for r in rows) for i in range(3)],
            "ranking": "lexicographic L,D,Q; compute time reported, not a tie breaker",
            "rows": rows}


def evaluate(directory, split, left, right, manifest, jobs=1):
    cases = read_json(Path(directory) / "data" / (split + ".json"))
    settings = dict(manifest["settings"], baseline_weights=manifest["baseline_weights"])
    work = [(str(Path(directory).resolve()), case, left, right, settings) for case in cases]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(work))) as pool:
            rows = list(pool.map(evaluate_case, work))
    else:
        rows = [evaluate_case(job) for job in work]
    return summarize(rows)
