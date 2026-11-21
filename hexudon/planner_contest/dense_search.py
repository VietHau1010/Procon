"""Bounded beam search for stock-rich maps: maximize distinct spot visits.

This is not used on stock-scarce maps where independent patrol routes compete
for portions. Whole-team decoding remains authoritative for fuel/meet timing.
"""

import time

from ..planner.solution import Meet
from .decoder import decode
from .objective import lexicographic_gain, end_value


def abundant(ctx):
    # Count-oriented independent patrol beams are a poor fit when most brands
    # are unique remote obligations. Keep the original solver on that family.
    repeated = sum(len(spots) > 1 for spots in ctx.config.spots_by_brand.values())
    return (ctx.n_patrols > 0 and bool(ctx.stocks)
            and all(s >= ctx.n_patrols for s in ctx.stocks)
            and repeated * 2 >= len(ctx.config.brands))


def beam_tours(ctx, slot, deadline=None, width=64, end_spot=None):
    """Open directed orienteering with a bitmask and bounded frontier per depth.

    Paths are proposed using step-shortest legs and REAL starting fuel. A
    tanker refill is not invented here; the decoder can exploit real meetings.
    States with the same (visited set,last) keep non-dominated time/fuel labels.
    """
    if width < 1:
        return []
    start, initial_fuel = ctx.start_of_patrol(slot), ctx.fuel_of_patrol(slot)
    cells = ctx.spot_cells
    n = len(cells)
    # Stop at most once per spot; zero-time starting-spot credit is consistent
    # with leaving a plain spot taking two steps (free collection at step 1).
    initial = ctx.config.spot_index_at.get(start)
    if initial is None:
        front = [(0, -1, 0.0, 0.0, ())]
    else:
        front = [(1 << initial, initial, 0.0, 0.0, (initial,))]
    distance = [[ctx.dist_step.steps_fuel(a, b) for b in cells] for a in cells]
    starts = [ctx.dist_step.steps_fuel(start, b) for b in cells]
    terminal = []
    for _ in range(n):
        if deadline is not None and time.monotonic() >= deadline:
            break
        expanded = {}
        for state_index, (mask, last, steps, fuel, tour) in enumerate(front):
            if (state_index % 32 == 0 and deadline is not None
                    and time.monotonic() >= deadline):
                break
            options = starts if last < 0 else distance[last]
            for nxt, (ds, df) in enumerate(options):
                if nxt == end_spot:
                    continue  # reserve the parking spot for the end of the day
                if mask & (1 << nxt) or steps+ds > ctx.steps or fuel+df > initial_fuel:
                    continue
                if end_spot is not None:
                    back_steps, back_fuel = distance[nxt][end_spot]
                    if steps+ds+back_steps > ctx.steps or fuel+df+back_fuel > initial_fuel:
                        continue
                candidate = (mask | (1 << nxt), nxt, steps+ds, fuel+df, tour+(nxt,))
                key = (candidate[0], nxt)
                labels = expanded.setdefault(key, [])
                if any(t <= candidate[2] and f <= candidate[3] for t, f, _ in labels):
                    continue
                labels[:] = [(t, f, s) for t, f, s in labels
                             if not (candidate[2] <= t and candidate[3] <= f)]
                labels.append((candidate[2], candidate[3], candidate))
        if not expanded:
            break
        ranked = [s for labels in expanded.values() for _, _, s in labels]
        ranked.sort(key=lambda s: (s[2], s[3], s[4]))
        # Retain a few alternatives for every depth. Different end spots and
        # brand coverage can beat the minimum-step route in the real simulator.
        if end_spot is None:
            by_end = {}
            for state in ranked:
                by_end.setdefault(state[1], state)
            terminal.extend(by_end.values())
        else:
            for mask, last, steps, fuel, tour in ranked[:min(16, width)]:
                back_steps, back_fuel = distance[last][end_spot]
                # The return may be a repeated starting spot: it preserves
                # parking, but does not count as another collectible serving.
                terminal.append((mask | (1 << end_spot), end_spot,
                                 steps+back_steps, fuel+back_fuel, tour+(end_spot,)))
        front = ranked[:width]
    terminal.sort(key=lambda s: (-s[0].bit_count(), s[2], s[3], s[4]))
    by_end = {}
    for state in terminal:
        by_end.setdefault(state[1], state)
    selected = list(by_end.values()) + terminal[:16]
    selected.sort(key=lambda s: (-s[0].bit_count(), s[2], s[3], s[4]))
    return [list(tour) for tour in dict.fromkeys(s[4] for s in selected)]


def replace_tour(solution, slot, tour):
    trial = solution.copy()
    old = trial.tours[slot]
    trial.tours[slot] = list(tour)
    mapped = []
    for m in trial.meets:
        if m.patrol != slot or m.stop_idx < 0:
            mapped.append(m)
        elif m.stop_idx < len(old) and old[m.stop_idx] in tour:
            mapped.append(Meet(slot, tour.index(old[m.stop_idx]), m.car))
    trial.meets = mapped
    waits = {}
    for (p, i), value in trial.waits.items():
        if p != slot or i < 0:
            waits[(p, i)] = value
        elif i < len(old) and old[i] in tour:
            waits[(p, tour.index(old[i]))] = value
    trial.waits = waits
    return trial


def improve_dense(ctx, solution, decoded, deadline=None, width=None):
    if not abundant(ctx):
        return solution, decoded
    if width is None:
        # Wider beams are worthwhile under a long contest window or an
        # iteration benchmark; don't starve ALNS on one/two-second days.
        width = 2048 if deadline is None or deadline-time.monotonic() >= 6 else 512
    best_sol, best = solution, decoded
    floor = end_value(ctx, decoded.result)
    for slot in range(ctx.n_patrols):
        if deadline is not None and time.monotonic() >= deadline:
            break
        # Keep a window for decoding; don't spend the entire remaining budget
        # generating an untested beam.
        beam_deadline = None
        if deadline is not None:
            remaining = max(0.0, deadline-time.monotonic())
            beam_deadline = time.monotonic()+remaining / max(2, 2*(ctx.n_patrols-slot))
        tours = beam_tours(ctx, slot, deadline=beam_deadline, width=width)
        for tour in tours:
            if deadline is not None and time.monotonic() >= deadline:
                break
            trial = replace_tour(best_sol, slot, tour)
            candidate = decode(ctx, trial)
            gain_q = candidate.result.servings - best.result.servings
            # A few extra moves consume fuel even if tomorrow's parking and
            # brand access are unchanged. Do not require exactly equal fuel;
            # protect a usable reserve and cap the terminal-value tradeoff.
            cfg_fuel = ctx.config.fuel_limits
            reserve = min(cfg_fuel, max(50, int(0.4*cfg_fuel)))
            fuel_safe = all(candidate.result.end_fuel[i] >= min(best.result.end_fuel[i], reserve)
                            for i in ctx.patrols) if candidate.result.ok else False
            if (candidate.result.ok
                    and lexicographic_gain(ctx, candidate.result) > lexicographic_gain(ctx, best.result)
                    and (ctx.days_left <= 0 or (fuel_safe and end_value(ctx, candidate.result)
                         >= floor - min(0.25*ctx.weights.B, max(0, gain_q)*ctx.weights.serving)))):
                best_sol, best = trial, candidate
    return best_sol, best
