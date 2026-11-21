"""Bounded, simulator-scored rescue of brands missing from the decoded plan."""

import time

from ..planner.solution import Meet
from .decoder import decode
from .objective import coverage_key, end_value


def _insert(solution, slot, index, spot, cut_tail=False):
    """Insert a stop without detaching surviving rendezvous/wait anchors."""
    trial = solution.copy()
    if cut_tail:
        trial.tours[slot] = trial.tours[slot][:index]
        trial.meets = [m for m in trial.meets
                       if m.patrol != slot or m.stop_idx < index]
        trial.waits = {(s, i): w for (s, i), w in trial.waits.items()
                       if s != slot or i < index}
    trial.tours[slot].insert(index, spot)
    trial.meets = [Meet(m.patrol, m.stop_idx + 1, m.car)
                   if m.patrol == slot and m.stop_idx >= index else m
                   for m in trial.meets]
    trial.waits = {((s, i + 1) if s == slot and i >= index else (s, i)): w
                   for (s, i), w in trial.waits.items()}
    return trial


def rescue_coverage(ctx, solution, decoded, deadline=None, max_evaluations=128):
    """Trade redundant route suffixes for real L/D, never for planned coverage.

    Try both preserving the suffix and dedicating its patrol to the missing
    brand. Full-team decoding checks fuel, tankers, stock competition and the
    types lost by the replacement. Before the last day, terminal value cannot
    decrease: grabbing an extra daily type is not worth stranding tomorrow.
    Search is deterministic and bounded by
    evaluations and the caller's deadline; unreachable/empty brands are skipped.
    """
    best_sol, best = solution, decoded
    min_terminal = end_value(ctx, decoded.result)
    remaining = max(0, max_evaluations)
    rounds = 0
    while remaining and rounds < len(ctx.config.brands):
        if deadline is not None and time.monotonic() >= deadline:
            break
        missing = []
        for brand in ctx.config.brands:
            if brand in best.result.brands:
                continue
            spots = [si for si in ctx.config.spots_by_brand[brand]
                     if ctx.stocks[si] > 0 and any(
                         ctx.dist_step.cost(ctx.start_of_patrol(p), ctx.spot_cells[si]) <= ctx.steps
                         for p in range(ctx.n_patrols))]
            if spots:
                missing.append((brand in ctx.visited_brands, len(spots), brand, spots))
        if not missing:
            break
        improved_sol, improved = best_sol, best
        for _, _, brand, spots in sorted(missing):
            if deadline is not None and time.monotonic() >= deadline:
                break
            proposals = []
            for spot in spots:
                target = ctx.spot_cells[spot]
                for slot, tour in enumerate(best_sol.tours):
                    start = ctx.start_of_patrol(slot)
                    if ctx.dist_step.cost(start, target) > ctx.steps:
                        continue
                    prefix = 0.0
                    prev = start
                    for index in range(len(tour) + 1):
                        travel = ctx.dist_step.cost(prev, target)
                        if prefix + travel <= ctx.steps:
                            extra = travel
                            if index < len(tour):
                                nxt = ctx.spot_cells[tour[index]]
                                extra += (ctx.dist_step.cost(target, nxt)
                                          - ctx.dist_step.cost(prev, nxt))
                            # Cost is only a proposal order. Decoder, not this
                            # distance estimate, decides what can be collected.
                            proposals.append((extra, slot, index, spot, False))
                            proposals.append((travel, slot, index, spot, True))
                        if index < len(tour):
                            nxt = ctx.spot_cells[tour[index]]
                            prefix += ctx.dist_step.cost(prev, nxt)
                            prev = nxt
            # Limit work per brand so one remote brand cannot exhaust the pass.
            per_brand = min(remaining, 32)
            for _, slot, index, spot, cut in sorted(proposals)[:per_brand]:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                trial = _insert(best_sol, slot, index, spot, cut_tail=cut)
                candidate = decode(ctx, trial)
                remaining -= 1
                if (coverage_key(ctx, candidate)[:2] > coverage_key(ctx, best)[:2]
                        and coverage_key(ctx, candidate) > coverage_key(ctx, improved)
                        and (ctx.days_left <= 0
                             or end_value(ctx, candidate.result) >= min_terminal)):
                    improved_sol, improved = trial, candidate
                if not remaining:
                    break
            if not remaining:
                break
        if improved is best:
            break
        best_sol, best = improved_sol, improved
        rounds += 1
    return best_sol, best


def rescue_servings(ctx, solution, decoded, deadline=None, max_evaluations=96):
    """Increase actual Q while holding actual L/D fixed.

    Proposals insert an unvisited spot with an already-covered brand into a
    patrol tour, or replace that tour's suffix. Full decoding checks stock,
    route truncation, fuel meets and incidental collections. Before the final
    day, require terminal value not to fall; this avoids spending tomorrow's
    fuel/position for today's extra portions.
    """
    target = coverage_key(ctx, decoded)[:2]
    best_sol, best = solution, decoded
    terminal_floor = end_value(ctx, decoded.result)
    remaining = max(0, max_evaluations)
    seen_trials = set()

    while remaining and (deadline is None or time.monotonic() < deadline):
        proposals = []
        collected_by_agent = {(agent, spot) for _, agent, spot in best.result.collections}
        for slot, tour in enumerate(best_sol.tours):
            start = ctx.start_of_patrol(slot)
            prefix, prev = 0.0, start
            in_tour = set(tour)
            agent = ctx.patrol_agent(slot)
            for index in range(len(tour) + 1):
                for spot, cell in enumerate(ctx.spot_cells):
                    brand = ctx.spot_brand[spot]
                    if (ctx.stocks[spot] <= 0 or brand not in best.result.brands
                            or (agent, spot) in collected_by_agent):
                        continue
                    # If the old route already planned this same stop but it
                    # was truncated, replace from its first occurrence rather
                    # than skip the only useful Q-rescue opportunity.
                    if spot in in_tour:
                        old_index = tour.index(spot)
                        if index <= old_index:
                            leg = ctx.dist_step.cost(prev, cell)
                            if prefix + leg <= ctx.steps:
                                proposals.append((leg, slot, index, spot, True))
                        continue
                    leg = ctx.dist_step.cost(prev, cell)
                    if prefix + leg > ctx.steps:
                        continue
                    extra = leg
                    if index < len(tour):
                        nxt = ctx.spot_cells[tour[index]]
                        extra += ctx.dist_step.cost(cell, nxt) - ctx.dist_step.cost(prev, nxt)
                    proposals.append((extra, slot, index, spot, False))
                    proposals.append((leg, slot, index, spot, True))
                if index < len(tour):
                    nxt = ctx.spot_cells[tour[index]]
                    prefix += ctx.dist_step.cost(prev, nxt)
                    prev = nxt

        if not proposals:
            break
        improved_sol, improved = best_sol, best
        for _, slot, index, spot, cut in sorted(proposals)[:remaining]:
            if deadline is not None and time.monotonic() >= deadline:
                break
            trial_key = (tuple(tuple(t) for t in best_sol.tours), slot, index, spot, cut)
            if trial_key in seen_trials:
                continue
            seen_trials.add(trial_key)
            trial = _insert(best_sol, slot, index, spot, cut_tail=cut)
            candidate = decode(ctx, trial)
            remaining -= 1
            key = coverage_key(ctx, candidate)
            if (key[:2] == target and key[2] > coverage_key(ctx, improved)[2]
                    and (ctx.days_left <= 0
                         or end_value(ctx, candidate.result) >= terminal_floor)):
                improved_sol, improved = trial, candidate
            if not remaining:
                break
        if improved is best:
            break
        best_sol, best = improved_sol, improved
    return best_sol, best
