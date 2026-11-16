"""Chọn vai trò xe tuần tra / xe tiếp nhiên liệu trước ngày 0 (Nâng cấp từ planner/roles.py).

Tích hợp công thức chọn nhanh thích ứng N in [3, 8] bảo đảm phản hồi tức thì,
không bao giờ lấn sang Ngày 0.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .. import rules
from ..model import AgentState, MatchConfig
from ..planner.roles import (
    RoleCandidate,
    candidates,
    centrality,
    rollout as base_rollout,
)
from ..planner.solution import Weights
from ..traffic import TrafficForecast


def choose_roles_for_match(num_agents: int, max_fuel: int, width: int, height: int) -> List[int]:
    """
    TẦNG 1: Công thức chọn số lượng xe Roles đầu trận (Thích ứng 3..8 xe):
    - N in [3, 6]: Dùng (N - 1) Patrol + 1 Refuel.
    - N in [7, 8]:
        + Nếu map <= 20x20 hoặc bình xăng >= 150: Dùng (N - 1) Patrol + 1 Refuel.
        + Nếu map lớn (>= 600 ô) và bình xăng nhỏ (< 120): Dùng (N - 2) Patrol + 2 Refuel.
    """
    if num_agents <= 6:
        return [rules.PATROL] * (num_agents - 1) + [rules.REFUEL]
    if width * height >= 600 and max_fuel < 120:
        return [rules.PATROL] * (num_agents - 2) + [rules.REFUEL, rules.REFUEL]
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
    """Chọn vai trò xe theo công thức tối ưu hoặc danh sách ứng viên."""
    log = log or (lambda msg: None)
    kinds = choose_roles_for_match(
        config.n_agents,
        config.fuel_limits,
        config.width,
        config.height,
    )
    opts = candidates(config)
    # Tìm candidate tương ứng
    matching = [c for c in opts if c.kinds == kinds]
    chosen = matching[0] if matching else RoleCandidate(kinds, f"{kinds.count(rules.REFUEL)} xe tiếp")
    chosen.evaluated = True
    if on_candidate is not None:
        on_candidate(dict(label=chosen.label, kinds=list(chosen.kinds), score=[0, 0, 0], phase="formula"))
    log(f"chọn vai trò (Tầng 1): {chosen.label} -> {kinds}")
    return kinds, opts
