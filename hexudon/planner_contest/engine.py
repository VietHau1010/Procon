"""Điều phối lập kế hoạch một ngày theo chiến thuật contest mới."""

import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..model import AgentState, MatchConfig
from ..pathing import DayGraph, DistTable, LegBook
from ..planner.solution import DayContext, DaySolution, Weights, build_context
from .alns import improve
from .construct import construct
from .decoder import Decoded, decode
from .objective import coverage_key, default_contest_weights
from .polish import fill_idle_suffixes
from .coverage import rescue_coverage, rescue_servings
from .dense_search import abundant, improve_dense


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
    tanker_order_search: bool = False,
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
    ctx = build_context(
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
    ctx.tanker_order_search = tanker_order_search
    return ctx


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
    enable_dense: bool = True,
    enable_lookahead: bool = False,
    on_lookahead: Optional[Callable[[dict], None]] = None,
) -> Decoded:
    """Lập kế hoạch một ngày: gửi bài an toàn tối ưu trước, rồi ALNS cải thiện tới hạn chót."""
    sol, first = safe_solution(ctx, seed)
    if on_plan is not None:
        on_plan(first)
    if deadline is not None and time.monotonic() >= deadline:
        return first
    best_seen = {"key": coverage_key(ctx, first)}

    def better(_sol, decoded: Decoded):
        if coverage_key(ctx, decoded) > best_seen["key"]:
            best_seen["key"] = coverage_key(ctx, decoded)
            if on_better is not None:
                on_better(decoded)
            if on_improve is not None:
                on_improve(decoded)

    # Reserve a small part of the search window for simulator-validated fill.
    work_deadline = deadline
    use_lookahead = enable_lookahead and ctx.days_left > 0
    if use_lookahead and deadline is not None:
        remaining = max(0.0, deadline - time.monotonic())
        use_lookahead = remaining >= 2.0
        if use_lookahead:
            work_deadline = deadline - min(3.0, remaining * 0.15)
    search_deadline = work_deadline
    if work_deadline is not None:
        remaining = max(0.0, work_deadline - time.monotonic())
        reserve = min(0.2, remaining * 0.1)
        if enable_dense and abundant(ctx):
            reserve = min(8.0, remaining * 0.2) if remaining >= 30 else min(3.0, remaining * 0.15)
        search_deadline = work_deadline - reserve
    result = improve(
        ctx,
        start=sol,
        deadline=search_deadline,
        max_iterations=max_iterations,
        seed=seed,
        on_better=better,
        on_progress=on_progress,
    )
    final = max((first, result.decoded), key=lambda d: coverage_key(ctx, d))
    final_solution = result.solution if final is result.decoded else sol
    final_solution, final = rescue_coverage(ctx, final_solution, final, deadline=work_deadline)
    if enable_dense:
        final_solution, final = improve_dense(ctx, final_solution, final, deadline=work_deadline)
    final_solution, higher_q = rescue_servings(ctx, final_solution, final, deadline=work_deadline)
    if coverage_key(ctx, higher_q) > coverage_key(ctx, final):
        final = higher_q
    final = fill_idle_suffixes(ctx, final, deadline=work_deadline)
    if use_lookahead:
        from .lookahead import choose_terminal_plan
        final, report = choose_terminal_plan(ctx, final, deadline=deadline, seed=seed)
        if on_lookahead is not None:
            on_lookahead(report)
    if on_plan is not None and final is not first:
        on_plan(final)
    return final
