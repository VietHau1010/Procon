"""Spend idle suffixes on one-way visits or excursions that return to parking."""

import time

from ..simulator import compact_plan, run_day
from .decoder import Decoded
from .objective import end_value, lexicographic_gain, score_day


def fill_idle_suffixes(ctx, decoded, deadline=None):
    """Replace final waits with validated visits, optionally returning home.

    The full simulator accounts for incidental collections, shared stocks and
    tanker encounters. A bounded terminal-value loss protects tomorrow's fuel
    and coverage. Departure fuel comes from the original trace, not end fuel:
    a tanker may have filled the patrol only at the end of its original wait.
    """
    best = decoded
    for _ in range(max(1, ctx.n_patrols * len(ctx.spot_cells))):
        if deadline is not None and time.monotonic() >= deadline:
            break
        candidates = []
        seen = {aid: set() for aid in ctx.patrols}
        for _, aid, si in best.result.collections:
            seen[aid].add(si)
        trace = []
        run_day(ctx.config, ctx.steps, ctx.agents, best.plan, ctx.stocks,
                ctx.road_status, trace=trace, strict_fuel=True)
        for aid in ctx.patrols:
            seq = best.plan[aid]
            if not seq or seq[-1] >= 0:
                continue
            spare = -seq[-1]
            src = best.result.end_pos[aid]
            departure = ctx.steps - spare
            # Leaving a starting spot at step zero may forfeit its free step-1
            # collection. Preserve it before spending the rest of the wait.
            delay = int(departure == 0 and src in ctx.config.spot_index_at
                        and ctx.config.spot_index_at[src] in seen[aid])
            available = spare - delay
            fuel = trace[departure + delay][2][aid]
            if not ctx.dist_step.has(src):
                continue
            for si, cell in enumerate(ctx.spot_cells):
                if si in seen[aid] or best.result.stocks_left[si] <= 0 or cell == src:
                    continue
                for leg in ctx.book.options(src, cell):
                    priority = (ctx.spot_brand[si] not in best.result.brands,
                                1.0 / max(1, leg.steps))
                    # Keep the incumbent one-way search. Its full simulation
                    # rejects any speculative use of fuel received too late.
                    if leg.steps <= spare and leg.fuel <= best.result.end_fuel[aid]:
                        candidates.append((priority, aid, 0, (leg,)))
                    # Returning retains tomorrow's free brand/serving and often
                    # keeps the original tanker meeting at the parking spot.
                    for back in ctx.book.options(cell, src):
                        if (leg.steps + back.steps <= available
                                and leg.fuel + back.fuel <= fuel):
                            candidates.append((priority, aid, delay, (leg, back)))
        candidates.sort(key=lambda row: row[0], reverse=True)
        improved = None
        before_value = end_value(ctx, best.result)
        for _, aid, delay, legs in candidates:
            if deadline is not None and time.monotonic() >= deadline:
                break
            plan = [list(seq) for seq in best.plan]
            spare = -plan[aid].pop()
            cur = best.result.end_pos[aid]
            if delay:
                plan[aid].append(-delay)
            used = delay
            for leg in legs:
                used += leg.steps
                for nxt in leg.path:
                    plan[aid].append(ctx.config.dir_index[cur][nxt])
                    cur = nxt
            if spare > used:
                plan[aid].append(-(spare - used))
            result = run_day(ctx.config, ctx.steps, ctx.agents, plan, ctx.stocks,
                             ctx.road_status, strict_fuel=True)
            if not result.ok or lexicographic_gain(ctx, result) <= lexicographic_gain(ctx, best.result):
                continue
            if end_value(ctx, result) < before_value - 0.25 * ctx.weights.B:
                continue
            if len(legs) > 1 and ctx.days_left > 0:
                # Extra Q must not sacrifice tomorrow's actual state: returning
                # to the same parking alone isn't sufficient if fuel is lower
                # or the excursion worsens a shared road for the next two days.
                if result.end_pos != best.result.end_pos:
                    continue
                if any(a < b for a, b in zip(result.end_fuel, best.result.end_fuel)):
                    continue
                if any(result.stay.get(c, 0) > best.result.stay.get(c, 0)
                       for c in ctx.config.road_cells):
                    continue
            candidate = Decoded(compact_plan(plan), result, score_day(ctx, result))
            if improved is None or (lexicographic_gain(ctx, result), candidate.score) > (
                    lexicographic_gain(ctx, improved.result), improved.score):
                improved = candidate
        if improved is None:
            break
        best = improved
    return best
