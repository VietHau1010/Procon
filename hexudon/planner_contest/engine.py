"""Điều phối lập kế hoạch một ngày theo chiến thuật contest mới."""

import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..model import AgentState, MatchConfig
from ..pathing import DayGraph, DistTable, LegBook
from ..planner.solution import DayContext, DaySolution, Weights, build_context
from .alns import improve
from .construct import construct
from .decoder import Decoded, decode
from .objective import default_contest_weights


def make_context(
    config: MatchConfig,
    day: int,
    agents: Sequence[AgentState],
    road_status: Dict[int, int],
    stocks: Sequence[int],
    visited_brands: Set[int],
    weights: Optional[Weights] = None,
    forecast_status: Optional[Dict[int, int]] = None,
    steps_override: Optional[int] = None,
) -> DayContext:
    """Dựng ngữ cảnh ngày, kèm các bảng khoảng cách và trọng số contest tối ưu."""
    graph = DayGraph(config, road_status)
    sources = sorted({s.pos for s in config.spots} | {a.pos for a in agents})
    dist_step = DistTable(graph, sources, "step")
    dist_fuel = DistTable(graph, sources, "fuel")
    book = LegBook(config, dist_step, dist_fuel)
    dist_next = None
    if forecast_status is not None and day + 1 < config.n_days:
        dist_next = DistTable(DayGraph(config, forecast_status), sources, "step")
    effective_weights = weights or default_contest_weights()
    return build_context(
        config=config,
        day=day,
        agents=agents,
        road_status=road_status,
        stocks=stocks,
        visited_brands=visited_brands,
        book=book,
        dist_step=dist_step,
        dist_next=dist_next,
        weights=effective_weights,
        steps_override=steps_override,
    )


def safe_solution(ctx: DayContext, seed: int = 0) -> Tuple[DaySolution, Decoded]:
    """Nghiệm an toàn: dựng bằng construct nâng cấp (Master Cycle + Brand Coverage + Hub Rendezvous + greedy fill)."""
    import random

    sol = construct(ctx, random.Random(seed))
    return sol, decode(ctx, sol)


def safe_plan(ctx: DayContext, seed: int = 0) -> Decoded:
    """Bài an toàn: gửi ngay cho server."""
    return safe_solution(ctx, seed)[1]


def plan_day(
    ctx: DayContext,
    deadline: Optional[float] = None,
    seed: int = 0,
    on_plan: Optional[Callable[[Decoded], None]] = None,
    max_iterations: int = 1_000_000,
    on_better: Optional[Callable[[Decoded], None]] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    on_improve: Optional[Callable[[Decoded], None]] = None,
) -> Decoded:
    """Lập kế hoạch một ngày: gửi bài an toàn tối ưu trước, rồi ALNS cải thiện tới hạn chót."""
    sol, first = safe_solution(ctx, seed)
    if on_plan is not None:
        on_plan(first)
    if deadline is not None and time.monotonic() >= deadline:
        return first
    best_seen = {"score": first.score}

    def better(_sol, decoded: Decoded):
        if decoded.score > best_seen["score"]:
            best_seen["score"] = decoded.score
            if on_better is not None:
                on_better(decoded)
            if on_improve is not None:
                on_improve(decoded)

    result = improve(
        ctx,
        start=sol,
        deadline=deadline,
        max_iterations=max_iterations,
        seed=seed,
        on_better=better,
        on_progress=on_progress,
    )
    final = result.decoded if result.decoded.score > first.score else first
    if on_plan is not None and final is not first:
        on_plan(final)
    return final
