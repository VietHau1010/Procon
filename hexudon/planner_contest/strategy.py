"""Bộ nhớ trận và điều phối chiến lược Contest (Master Cycle TSP + ALNS + Multi-Tier Optimization)."""

import time
from typing import Callable, Dict, List, Optional, Sequence, Set

from .. import rules
from ..model import AgentState, DayState, MatchConfig
from ..planner.strategy import (
    DEFAULT_NET_MARGIN,
    DEFAULT_ROLE_SECONDS,
    DEFAULT_SEARCH_SECONDS,
    MAX_SUBMITS_PER_DAY,
    RESUBMIT_MIN_GAIN,
    MatchStrategist as BaseMatchStrategist,
)
from ..planner.solution import Weights
from ..traffic import TrafficForecast
from .decoder import Decoded
from .engine import make_context, plan_day
from .objective import default_contest_weights, lexicographic_gain
from .roles import choose_roles_for_match

_ORIGINAL_BASE_SELECT_ROLES = BaseMatchStrategist.select_roles


class MatchStrategist(BaseMatchStrategist):
    """Strategist sử dụng ALNS tối ưu hoá theo 5 tầng chiến thuật contest."""

    def __init__(
        self,
        config: MatchConfig,
        weights: Optional[Weights] = None,
        search_seconds: Optional[float] = DEFAULT_SEARCH_SECONDS,
        net_margin: float = DEFAULT_NET_MARGIN,
        role_seconds: Optional[float] = DEFAULT_ROLE_SECONDS,
        seed: int = 0,
        log: Optional[Callable[[str], None]] = None,
        emit: Optional[Callable[[str, dict], None]] = None,
        search_iterations: Optional[int] = None,
        role_iterations: Optional[int] = None,
    ):
        contest_w = weights or default_contest_weights()
        super().__init__(
            config=config,
            weights=contest_w,
            search_seconds=search_seconds,
            net_margin=net_margin,
            role_seconds=role_seconds,
            seed=seed,
            log=log,
            emit=emit,
            search_iterations=search_iterations,
            role_iterations=role_iterations,
        )

    def select_roles(
        self, budget_seconds: Optional[float] = None, cap_by_start: bool = True
    ) -> List[int]:
        """TẦNG 1: Chọn vai trò tức thì theo công thức thích ứng choose_roles_for_match.

        Không cần chạy mô phỏng tốn hàng giây, bảo đảm không lấn sang Ngày 0.
        """
        if BaseMatchStrategist.select_roles is not _ORIGINAL_BASE_SELECT_ROLES:
            return super().select_roles(budget_seconds=budget_seconds, cap_by_start=cap_by_start)

        kinds = choose_roles_for_match(
            self.config.n_agents,
            self.config.fuel_limits,
            self.config.width,
            self.config.height,
        )
        self.kinds = kinds
        self.forecast = TrafficForecast(
            self.config,
            n_patrols=max(1, sum(1 for k in kinds if k == rules.PATROL)),
            pop=self.forecast.pop,
        )
        n_cars = sum(1 for k in kinds if k == rules.REFUEL)
        self.log(
            f"chọn vai trò contest (Tầng 1): {n_cars} xe tiếp nhiên liệu / {len(kinds)} agent -> {kinds}"
        )
        if self.emit is not None:
            self.emit("roles_chosen", {"kinds": list(kinds)})
        return kinds

    def plan_day(
        self,
        state: DayState,
        deadline: Optional[float] = None,
        submit: Optional[Callable[[List[List[int]]], bool]] = None,
        steps_override: Optional[int] = None,
        record: bool = True,
    ) -> Decoded:
        cfg = self.config
        started = time.monotonic()
        self.reconcile(state)

        agents = [AgentState(a.kind, a.pos, a.fuel) for a in state.agents]
        server_kinds = [a.kind for a in state.agents]
        if server_kinds and server_kinds != self.kinds[: len(server_kinds)]:
            self.log(f"lệch vai trò: ta {self.kinds}, server {server_kinds}; theo server")
            self.kinds = list(server_kinds)

        status = dict(state.road_status)
        forecast_status = self.forecast.predict(
            status,
            self.status_yesterday,
            own_stay_yesterday=self.own_stay_yesterday,
            floor_today=state.day == 1,
        )
        ctx = make_context(
            cfg,
            state.day,
            agents,
            status,
            [s.stocks for s in cfg.spots],
            self.visited_brands,
            weights=self.weights,
            forecast_status=forecast_status,
            steps_override=steps_override,
        )

        if deadline is None and (self.search_iterations is None or state.ends_at is not None):
            deadline = time.monotonic() + self.budget_for(state)
        max_iterations = (
            self.search_iterations if self.search_iterations is not None else 1_000_000
        )

        sent = {"count": 0, "decoded": None, "gain": None}
        self.last_sent = None
        on_safe, hooks = self._watch(state, agents, status, ctx.steps, started)

        def offer(decoded: Decoded, final: bool) -> None:
            if submit is None:
                return
            limit = MAX_SUBMITS_PER_DAY if final else MAX_SUBMITS_PER_DAY - 1
            if sent["count"] >= limit:
                return
            gain = lexicographic_gain(ctx, decoded.result)
            last = sent["decoded"]
            if last is not None and not (
                gain > sent["gain"]
                or decoded.score >= last.score + RESUBMIT_MIN_GAIN * ctx.weights.B
            ):
                return
            sent["count"] += 1
            if submit(decoded.plan):
                sent["decoded"] = decoded
                sent["gain"] = gain
                self.last_sent = decoded

        def on_plan(decoded: Decoded):
            nonlocal on_safe
            if on_safe is not None:
                on_safe(decoded)
                on_safe = None
            offer(decoded, final=True)

        best = plan_day(
            ctx,
            deadline=deadline,
            seed=self.seed,
            on_plan=on_plan,
            on_improve=lambda decoded: offer(decoded, final=False),
            max_iterations=max_iterations,
            **hooks,
        )
        applied = best if submit is None or sent["decoded"] is None else sent["decoded"]
        if record:
            self.record(state, applied, status)
        return applied
