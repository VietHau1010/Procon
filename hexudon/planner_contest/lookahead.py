"""Experimental, bounded next-day evaluation of today's idle-suffix parking.

This is a projection, not knowledge of opponents' future routes. The feature
is opt-in and must pass full-match tests before becoming a default policy.
"""

import time

from .. import rules
from ..model import AgentState
from ..pathing import DistTable, LegBook
from ..simulator import compact_plan, run_day
from .decoder import Decoded
from .objective import end_value, lexicographic_gain, score_day


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def parking_candidates(ctx, incumbent, deadline=None, max_checks=32):
    """Change only a final wait; preserve every collection already obtained.

    Tankers may finish next to a low-fuel patrol, and patrols may finish at a
    different nearby shop. All fuel/tanker interactions are validated together
    using the actual fuel at departure, not a speculative later refill.
    """
    if _expired(deadline) or max_checks <= 0:
        return []
    trace = []
    original = run_day(ctx.config, ctx.steps, ctx.agents, incumbent.plan,
                       ctx.stocks, ctx.road_status, trace=trace, strict_fuel=True)
    if not original.ok:
        return []
    original_collections = {(aid, spot) for _, aid, spot in original.collections}
    sources = sorted(set(original.end_pos))
    steps = DistTable(ctx.dist_step.graph, sources, "step")
    fuel = DistTable(ctx.dist_step.graph, sources, "fuel")
    book = LegBook(ctx.config, steps, fuel)
    proposals, seen, checks = [], {tuple(tuple(s) for s in incumbent.plan)}, 0
    # Test tanker parking first; this preserves the single-tanker assignment.
    for aid in list(ctx.cars) + list(ctx.patrols):
        if _expired(deadline):
            break
        seq = incumbent.plan[aid]
        if not seq or seq[-1] >= 0:
            continue
        spare, src = -seq[-1], original.end_pos[aid]
        departure = ctx.steps - spare
        collected_at_start = any(a == aid and ctx.spot_cells[s] == src
                                 for _, a, s in original.collections)
        delay = int(departure == 0 and collected_at_start)
        available = spare - delay
        if available <= 0:
            continue
        departure_fuel = trace[departure + delay][2][aid]
        targets = sorted(set(ctx.spot_cells), key=lambda c: (steps.cost(src, c), c))
        if aid in ctx.cars:
            rendezvous = [original.end_pos[p] for p in
                          sorted(ctx.patrols, key=lambda p: (original.end_fuel[p], p))]
            targets = rendezvous + targets
        targets = list(dict.fromkeys(c for c in targets if c != src))
        feasible_targets = 0
        for target in targets:
            if feasible_targets >= 4 or checks >= max_checks or _expired(deadline):
                break
            options = [leg for leg in book.options(src, target)
                       if leg.steps <= available and (aid in ctx.cars or leg.fuel <= departure_fuel)]
            if not options:
                continue
            feasible_targets += 1
            for leg in options:
                if checks >= max_checks or _expired(deadline):
                    break
                plan = [list(s) for s in incumbent.plan]
                suffix = [-delay] if delay else []
                cell = src
                for nxt in leg.path:
                    suffix.append(ctx.config.dir_index[cell][nxt])
                    cell = nxt
                left = available - leg.steps
                if left:
                    suffix.append(-left)
                plan[aid] = plan[aid][:-1] + suffix
                plan = compact_plan(plan)
                signature = tuple(tuple(s) for s in plan)
                if signature in seen:
                    continue
                seen.add(signature)
                checks += 1
                result = run_day(ctx.config, ctx.steps, ctx.agents, plan,
                                 ctx.stocks, ctx.road_status, strict_fuel=True)
                if not result.ok:
                    continue
                collected = {(a, s) for _, a, s in result.collections}
                if not original_collections.issubset(collected):
                    continue
                if lexicographic_gain(ctx, result) < lexicographic_gain(ctx, original):
                    continue
                if result.end_pos == original.end_pos and result.end_fuel == original.end_fuel:
                    continue
                proposals.append((aid, Decoded(plan, result, score_day(ctx, result))))
    proposals.sort(key=lambda row: (lexicographic_gain(ctx, row[1].result),
                                    end_value(ctx, row[1].result)), reverse=True)
    # Keep actor diversity so a high parking heuristic cannot hide the tanker.
    diverse, remaining, actors = [], [], set()
    for aid, decoded in proposals:
        if aid not in actors:
            diverse.append(decoded)
            actors.add(aid)
        else:
            remaining.append(decoded)
    return diverse + remaining


def projected_roads(ctx, result):
    """Use the existing forecast plus a lower bound from candidate own traffic.

    Opponents' next routes are unknown. Do not treat this projection as the
    server's actual future road state or assume the player's traffic vanishes.
    """
    graph = ctx.dist_next.graph if ctx.dist_next is not None else ctx.dist_step.graph
    return {cell: max(rules.STEP_ROAD.index(graph.step_cost[cell]),
                      rules.road_status_for(result.stay.get(cell, 0) / max(1, ctx.config.players),
                                            ctx.config.busy_threshold, ctx.config.jammed_threshold))
            for cell in ctx.config.road_cells}


def project_next_day(ctx, decoded, seed, iterations):
    """Equal deterministic search for all projections; never recurse lookahead."""
    from .engine import make_context, plan_day
    agents = [AgentState(agent.kind, pos, fuel)
              for agent, pos, fuel in zip(ctx.agents, decoded.result.end_pos, decoded.result.end_fuel)]
    next_ctx = make_context(ctx.config, ctx.day + 1, agents, projected_roads(ctx, decoded.result),
                            [s.stocks for s in ctx.config.spots],
                            ctx.visited_brands | decoded.result.brands, weights=ctx.weights)
    return plan_day(next_ctx, seed=seed, max_iterations=iterations,
                    enable_dense=False, enable_lookahead=False)


def two_day_key(ctx, today, tomorrow):
    combined = today.result.brands | tomorrow.result.brands
    return (len(combined - ctx.visited_brands),
            len(today.result.brands) + len(tomorrow.result.brands),
            today.result.servings + tomorrow.result.servings)


def choose_terminal_plan(ctx, incumbent, deadline=None, seed=0, iterations=8,
                         max_candidates=3):
    """Compare incumbent and up to three validated alternatives over two days.

    Never lower today's L/D/Q or lose any collected brand. If the entire
    nominated comparison cannot finish before the deadline, keep the incumbent.
    Forecast-based next-day gains still require full-match empirical validation.
    """
    report = {"phase": "lookahead", "selected": False, "proposed": 0,
              "rollouts": 0, "projection": "forecast; not server future state"}
    if (ctx.days_left <= 0 or ctx.n_cars != 1 or not incumbent.result.ok
            or _expired(deadline) or max_candidates <= 0):
        report["status"] = "skipped"
        return incumbent, report
    candidates = parking_candidates(ctx, incumbent, deadline=deadline)[:max_candidates]
    report["proposed"] = len(candidates)
    if not candidates or _expired(deadline):
        report["status"] = "no_candidates" if not candidates else "deadline"
        return incumbent, report
    projected = []
    for decoded in [incumbent] + candidates:
        if _expired(deadline):
            report["status"] = "deadline"
            return incumbent, report
        tomorrow = project_next_day(ctx, decoded, seed=seed, iterations=max(0, iterations))
        report["rollouts"] += 1
        if _expired(deadline) or not tomorrow.result.ok:
            report["status"] = "deadline" if _expired(deadline) else "invalid_projection"
            return incumbent, report
        projected.append(two_day_key(ctx, decoded, tomorrow))
    best_index = max(range(len(projected)), key=projected.__getitem__)
    report.update(status="complete", baseline=list(projected[0]), best=list(projected[best_index]))
    if best_index and projected[best_index] > projected[0]:
        report["selected"] = True
        return candidates[best_index - 1], report
    return incumbent, report
