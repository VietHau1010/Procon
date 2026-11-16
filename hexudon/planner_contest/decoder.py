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
from .objective import score_day


def decode(ctx: DayContext, sol: DaySolution) -> Decoded:
    """Giải mã nghiệm thành kế hoạch hợp lệ kèm điểm tính theo hàm mục tiêu contest."""
    plan, truncated, dropped = build_plan(ctx, sol)
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
