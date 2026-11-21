"""Bộ giải mã cho chiến thuật contest: DaySolution -> kế hoạch hợp lệ kèm điểm contest."""

from typing import List

from ..planner.decoder import (
    Decoded,
    PatrolWalk,
    build_plan,
    compact_plan,
    repair_to_valid,
    walk_patrol,
)
from ..planner.solution import DayContext, DaySolution
from .objective import coverage_key, score_day


def _decode(ctx: DayContext, sol: DaySolution, car_scheduler=None) -> Decoded:
    if car_scheduler is None:
        plan, truncated, dropped = build_plan(ctx, sol)
    else:
        plan, truncated, dropped = build_plan(ctx, sol, car_scheduler=car_scheduler)
    plan, result, cuts = repair_to_valid(ctx, plan)
    plan = compact_plan(plan)
    if not result.ok:
        return Decoded(
            plan=plan,
            result=result,
            score=float("-inf"),
            truncated=truncated + cuts,
            dropped_meets=dropped,
        )
    score = score_day(ctx, result, truncated=truncated + cuts, dropped_meets=dropped)
    return Decoded(
        plan=plan,
        result=result,
        score=score,
        truncated=truncated + cuts,
        dropped_meets=dropped,
    )


def decode(ctx: DayContext, sol: DaySolution) -> Decoded:
    """Keep the old decoder unless an opt-in tanker order scores better.

    More meetings alone are not a gain. Compare real whole-team collections,
    preserve every brand the fallback obtained, and leave equal LDQ unchanged.
    This protects today's result, not an unproven whole-match improvement.
    """
    baseline = _decode(ctx, sol)
    if (not getattr(ctx, "tanker_order_search", False) or ctx.n_cars != 1
            or baseline.dropped_meets == 0 or not sol.meets):
        return baseline
    from .tanker import scheduler
    best = baseline
    for mode in ("deadline", "beam"):
        candidate = _decode(ctx, sol, car_scheduler=scheduler(mode))
        if (candidate.result.ok and baseline.result.brands.issubset(candidate.result.brands)
                and coverage_key(ctx, candidate)[:3] > coverage_key(ctx, best)[:3]):
            best = candidate
    return best
