"""Bounded directed-route shortcuts, followed by simulator-verified insertion."""

import math
import time

from ..planner.solution import Meet
from .construct import greedy_fill
from .decoder import decode
from .objective import coverage_key, end_value, lexicographic_gain


def reorder(solution, slot, order):
    """order contains OLD indices: keep rendezvous and waits attached to stops.

    Occurrence indices, rather than spot IDs, distinguish duplicate visits.
    """
    old = solution.tours[slot]
    if sorted(order) != list(range(len(old))):
        raise ValueError("Route permutation phải chứa mỗi chỉ số đúng một lần")
    result = solution.copy()
    result.tours[slot] = [old[index] for index in order]
    inverse = {old_index: new_index for new_index, old_index in enumerate(order)}
    result.meets = [Meet(m.patrol, inverse[m.stop_idx], m.car)
                    if m.patrol == slot and m.stop_idx >= 0 else m for m in result.meets]
    result.waits = {((s, inverse[i]) if s == slot and i >= 0 else (s, i)): value
                    for (s, i), value in result.waits.items()}
    return result


def _cost(ctx, slot, tour):
    cur, steps, fuel = ctx.start_of_patrol(slot), 0.0, 0.0
    for spot in tour:
        cell = ctx.spot_cells[spot]
        s, f = ctx.dist_step.steps_fuel(cur, cell)
        steps += s
        fuel += f
        cur = cell
    return steps, fuel


def shorten_route(ctx, solution, slot, deadline=None, max_moves=12):
    """Best-improvement 2-opt AND one-stop relocation on an open directed tour.

    Vertex-weighted paths are asymmetric. Reversing only the four boundary
    edges is not a valid delta; evaluate the complete directed route instead.
    Neither the end spot nor route fuel can worsen in this phase.
    """
    current = solution
    for _ in range(max_moves):
        if deadline is not None and time.monotonic() >= deadline:
            break
        tour = current.tours[slot]
        n = len(tour)
        if n < 3:
            break
        cost, fuel = _cost(ctx, slot, tour)
        if not math.isfinite(cost) or not math.isfinite(fuel):
            break
        best, best_cost = None, cost
        # Preserve the final stop for tomorrow's free collection/parking value.
        for i in range(n-1):
            if deadline is not None and time.monotonic() >= deadline:
                break
            for j in range(i+1, n-1):
                order = list(range(n))
                order[i:j+1] = reversed(order[i:j+1])
                candidate = [tour[k] for k in order]
                distance, need = _cost(ctx, slot, candidate)
                if distance < best_cost and need <= fuel:
                    best, best_cost = order, distance
            for j in range(n-1):
                if i == j:
                    continue
                order = list(range(n))
                value = order.pop(i)
                order.insert(j, value)
                candidate = [tour[k] for k in order]
                distance, need = _cost(ctx, slot, candidate)
                if distance < best_cost and need <= fuel:
                    best, best_cost = order, distance
        if best is None:
            break
        current = reorder(current, slot, best)
    return current


def shortcut_and_fill(ctx, solution, decoded, deadline=None, max_passes=2):
    """Return the unchanged incumbent unless real L/D/Q increase safely.

    Keep the original end-value floor across the entire pass. Any early tanker
    arrival/wait/stock effect is checked with whole-team decoding. Gains today
    still do not prove higher full-match scores: terminal value is a heuristic.
    """
    best_sol, best = solution, decoded
    terminal_floor = end_value(ctx, decoded.result)
    q_bound = sum(min(max(0, stock), ctx.n_patrols) for stock in ctx.stocks)
    if best.result.servings >= q_bound:
        return best_sol, best
    for _ in range(max_passes):
        changed = False
        for slot in range(ctx.n_patrols):
            if deadline is not None and time.monotonic() >= deadline:
                return best_sol, best
            trial = shorten_route(ctx, best_sol, slot, deadline=deadline)
            if trial is best_sol:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return best_sol, best
            # Insertion is bounded; decode can reject speculative fuel assumed
            # by greedy construction or lost tanker meets after reordering.
            filled = greedy_fill(ctx, trial, max_inserts=8)
            candidate = decode(ctx, filled)
            if (candidate.result.ok
                    and lexicographic_gain(ctx, candidate.result) > lexicographic_gain(ctx, best.result)
                    and (ctx.days_left <= 0 or end_value(ctx, candidate.result) >= terminal_floor)):
                best_sol, best = filled, candidate
                changed = True
        if not changed:
            break
    return best_sol, best
