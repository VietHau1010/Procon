"""Chọn vai trò bằng rollout contest; công thức trung tâm là phương án dự phòng."""

import time
from typing import Callable, List, Optional, Sequence, Tuple

from .. import rules
from ..model import AgentState, MatchConfig
from ..simulator import road_status_from_history
from ..planner.roles import (
    RoleCandidate,
    candidates,
    centrality,
)
from ..planner.solution import Weights
from ..traffic import TrafficForecast


def choose_roles_for_match(
    config_or_num_agents,
    max_fuel: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[int]:
    """
    TẦNG 1: Công thức chọn số lượng xe Roles đầu trận (Thích ứng 3..8 xe):
    - N in [3, 6]: Dùng (N - 1) Patrol + 1 Refuel.
    - Mọi map: đúng một xe tiếp nhiên liệu; rollout chọn xe xuất phát phù hợp nhất.
    - Luôn đặt xe Refuel tại agent có vị trí xuất phát trung tâm nhất (theo centrality(config)).
    """
    if isinstance(config_or_num_agents, MatchConfig):
        cfg = config_or_num_agents
        cen = centrality(cfg)
        order = sorted(range(cfg.n_agents), key=lambda i: cen[i])
        n_refuel = 1
        roles = [rules.PATROL] * cfg.n_agents
        for idx in order[:n_refuel]:
            roles[idx] = rules.REFUEL
        return roles
    else:
        num_agents = config_or_num_agents
        if num_agents <= 6:
            return [rules.PATROL] * (num_agents - 1) + [rules.REFUEL]
        return [rules.PATROL] * (num_agents - 1) + [rules.REFUEL]


def select_roles(
    config: MatchConfig,
    budget_seconds: float = 3.0,
    weights: Optional[Weights] = None,
    seed: int = 0,
    log: Optional[Callable[[str], None]] = None,
    on_candidate: Optional[Callable[[dict], None]] = None,
    iterations_per_day: Optional[int] = None,
) -> Tuple[List[int], List[RoleCandidate]]:
    """Chọn vai trò bằng rollout của chính planner contest.

    Centrality chỉ tạo ứng viên. Quyết định cuối cùng phải nhìn thấy toàn bộ
    trận, vì một xe tiếp ở vị trí trung tâm theo khoảng cách hình học chưa chắc
    là xe tiếp có thể cứu được các tour bị thiếu xăng ở các ngày ngắn.
    """
    log = log or (lambda msg: None)
    started = time.monotonic()
    fallback = choose_roles_for_match(config)
    if budget_seconds <= 0 and iterations_per_day is None:
        return fallback, [RoleCandidate(fallback, "công thức dự phòng")]

    hard_deadline = (
        None if iterations_per_day is not None
        else started + max(0.0, budget_seconds)
    )
    # Exactly one tanker; shortlist the central starts within the role window.
    # Keep the legacy planner's candidate generator unchanged.
    opts = [c for c in candidates(config)
            if sum(kind == rules.REFUEL for kind in c.kinds) == 1]
    if not any(c.kinds == fallback for c in opts):
        opts.append(RoleCandidate(fallback, "1 xe tiếp (dự phòng)"))
    opts.sort(key=lambda c: c.kinds != fallback)

    def tell(cand: RoleCandidate, phase: str, score, days: int) -> None:
        cand.score = tuple(score)
        cand.evaluated = days == config.n_days
        if on_candidate is not None:
            on_candidate(dict(label=cand.label, kinds=list(cand.kinds), score=list(score),
                              phase=phase, days=days))

    # First pass is cheap and uses the same construct/decoder as live planning.
    for cand in opts:
        if hard_deadline is not None and time.monotonic() >= hard_deadline:
            break
        score, days = _rollout_contest(
            config, cand.kinds, weights=weights, seed=seed,
            # Deterministic training/benchmark calls must also cap the first
            # role pass. Previously role_iterations only limited the two
            # finalists; the initial candidates silently used safe_solution()
            # with an unbounded search and made every training case appear to
            # hang.
            deadline=hard_deadline,
            iterations_per_day=iterations_per_day,
        )
        tell(cand, "contest-rollout", score, days)
        log(f"vai trò {cand.label}: L={score[0]} D={score[1]} Q={score[2]} ({days}/{config.n_days} ngày)")

    ranked = sorted((c for c in opts if c.evaluated), key=lambda c: c.score, reverse=True)
    if not ranked:
        return fallback, opts

    # Spend remaining deterministic/real-time budget on the two best role sets.
    finalists = ranked[:2]
    refined_scores = {}
    if iterations_per_day is not None:
        for cand in finalists:
            cand.refined, days = _rollout_contest(
                config, cand.kinds, weights=weights, seed=seed,
                deadline=None, iterations_per_day=iterations_per_day,
            )
            if days == config.n_days:
                refined_scores[tuple(cand.kinds)] = cand.refined
            if on_candidate is not None:
                on_candidate(dict(label=cand.label, kinds=list(cand.kinds),
                                  score=list(cand.refined), phase="contest-refine", days=days))
    else:
        remaining = (hard_deadline - time.monotonic()) if hard_deadline else 0
        if remaining > 0.15 and len(finalists) > 1:
            for index, cand in enumerate(finalists):
                if hard_deadline is not None and time.monotonic() >= hard_deadline:
                    break
                slice_deadline = time.monotonic() + remaining / (len(finalists) - index)
                if hard_deadline is not None:
                    slice_deadline = min(slice_deadline, hard_deadline)
                cand.refined, days = _rollout_contest(
                    config, cand.kinds, weights=weights, seed=seed,
                    deadline=slice_deadline, iterations_per_day=8,
                )
                if days == config.n_days:
                    refined_scores[tuple(cand.kinds)] = cand.refined
                if on_candidate is not None:
                    on_candidate(dict(label=cand.label, kinds=list(cand.kinds),
                                      score=list(cand.refined), phase="contest-refine", days=days))
                remaining = max(0.0, (hard_deadline - time.monotonic()) if hard_deadline else 0)

    # Compare like with like. Partial refinement (or just one complete finalist)
    # cannot displace the winner of the complete first round.
    chosen = (max(finalists, key=lambda c: refined_scores[tuple(c.kinds)])
              if len(refined_scores) == len(finalists) else ranked[0])

    log(f"chọn vai trò contest: {chosen.label} -> {chosen.kinds}")
    return list(chosen.kinds), opts


def _rollout_contest(
    config: MatchConfig,
    kinds: Sequence[int],
    weights: Optional[Weights],
    seed: int,
    deadline: Optional[float],
    iterations_per_day: Optional[int],
) -> Tuple[Tuple[int, int, int], int]:
    """Carry position/fuel across days; reset stocks each day, as in the simulator.

    Traffic here is our own contribution divided by the configured team count;
    opponent routes are unknown. Forecasting is used for terminal valuation.
    """
    from .engine import make_context, plan_day, safe_solution

    agents = [AgentState(kinds[i], config.agent_starts[i], config.fuel_limits)
              for i in range(config.n_agents)]
    visited = set()
    daily_types = 0
    servings = 0
    status = {c: rules.SMOOTH for c in config.road_cells}
    history = []
    prev_status = None
    prev_stay = None
    forecast = TrafficForecast(config, n_patrols=max(1, sum(k == rules.PATROL for k in kinds)))
    days_done = 0

    for day in range(config.n_days):
        if deadline is not None and time.monotonic() >= deadline:
            break
        next_status = forecast.predict(
            status, prev_status, own_stay_yesterday=prev_stay,
            floor_today=day == 1,
        )
        ctx = make_context(
            config, day, agents, status, [s.stocks for s in config.spots], visited,
            weights=weights, forecast_status=next_status,
        )
        if deadline is not None and time.monotonic() >= deadline:
            break
        if iterations_per_day is None:
            _, decoded = safe_solution(ctx, seed=seed + day)
        else:
            day_deadline = None if deadline is None else min(
                deadline, time.monotonic() + 0.25
            )
            decoded = plan_day(
                ctx, deadline=day_deadline, seed=seed + day,
                max_iterations=iterations_per_day,
                enable_dense=False,
            )
        result = decoded.result
        if not result.ok or (deadline is not None and time.monotonic() >= deadline):
            break
        visited |= result.brands
        daily_types += len(result.brands)
        servings += result.servings
        agents = [AgentState(kinds[i], result.end_pos[i], result.end_fuel[i])
                  for i in range(config.n_agents)]
        old_status = status
        history.append(result.stay)
        status = road_status_from_history(config, history)
        prev_status = old_status
        prev_stay = result.stay
        days_done += 1
    return (len(visited), daily_types, servings), days_done
