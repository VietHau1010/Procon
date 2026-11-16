"""Hàm mục tiêu chuẩn cho giải đấu Procon:
Tuân thủ tuyệt đối thứ tự từ điển: L (tổng nhãn toàn giải) > D (tổng nhãn mỗi ngày) > Q (tổng khẩu phần udon).

Cải tiến then chốt:
1. Đặt w.serving = 20.0, w.fuel_bonus = 0.01: Nhiên liệu chỉ dùng để phân định thắng thua khi
   điểm khẩu phần bằng nhau, tuyệt đối không bao giờ hi sinh udon để tiết kiệm xăng.
2. Fuel Safety Penalty: Phạt nặng xe tuần tra kết thúc ngày với lượng xăng < 50 mà không có xe
   tiếp nhiên liệu đứng kèm, ngăn chặn triệt để thảm hoạ kẹt xăng qua ngày hôm sau.
3. Dead-end Clumping Penalty: Phạt dồn nhiều hơn 1 xe vào cụm ngõ cụt xa {5, 34, 40, 142, 197, 233}.
4. End-on-Spot Bonus: Thưởng đỗ xe cuối ngày tại ô quán ăn để ngày hôm sau tự động ăn điểm ở bước 1.
"""

from typing import Dict, List, Sequence

from .. import rules
from ..simulator import DayResult
from ..planner.solution import DayContext, Weights

INF = float("inf")


def default_contest_weights() -> Weights:
    """Bộ trọng số contest cân bằng tối ưu."""
    return Weights(
        B=1000.0,
        new_brand=0.5,
        serving=20.0,         # Tăng từ 1.0 lên 20.0 để ăn udon được ưu tiên tối đa
        cover_penalty=1000.0,
        urgency=0.6,
        end_on_spot=0.5,      # Đỗ trên spot để bước 1 ngày mai ăn udon miễn phí
        near_chain=0.4,
        strand=2.0,           # Phạt mạnh kẹt xăng qua ngày
        fuel_bonus=0.01,      # Giảm từ 1.0 xuống 0.01 để xăng không cạnh tranh với udon
        road_park=0.05,
        diag_penalty=0.01,
    )


def _dist_to_brand(ctx: DayContext, cell: int, brand: int) -> float:
    table = ctx.dist_next or ctx.dist_step
    best = INF
    for si in ctx.config.spots_by_brand.get(brand, ()):
        d = table.cost(cell, ctx.spot_cells[si])
        if d < best:
            best = d
    return best


def end_value(ctx: DayContext, result: DayResult) -> float:
    """Giá trị cuối ngày V: bảo đảm an toàn xăng cho ngày mai và chống kẹt xe ngõ cụt."""
    if ctx.days_left <= 0:
        return 0.0
    w = ctx.weights
    B = w.B
    cfg = ctx.config
    value = 0.0

    end_pos = result.end_pos
    end_fuel = result.end_fuel
    patrol_cells = [end_pos[i] for i in ctx.patrols]
    car_cells = [end_pos[i] for i in ctx.cars]

    steps_rest = sum(cfg.day_steps[ctx.day + 1:])
    unvisited = [b for b in cfg.brands if b not in ctx.visited_brands and b not in result.brands]

    table = ctx.dist_next or ctx.dist_step

    # V_cover + V_urgency: Phạt nếu không thể ghé các nhãn hàng còn thiếu trong phần trận còn lại
    for brand in unvisited:
        best = min((table.cost(c, ctx.spot_cells[si]) for c in patrol_cells for si in cfg.spots_by_brand.get(brand, ())), default=INF)
        if best == INF or best > steps_rest:
            value -= w.cover_penalty * B
        else:
            value -= w.urgency * B * min(1.0, best / max(1, steps_rest))

    # V_spot: kết thúc trên spot thì hôm sau thu miễn phí ở bước 1
    spot_at = cfg.spot_index_at
    for c in patrol_cells:
        if c in spot_at:
            value += w.end_on_spot * B + w.serving

    # V_deadend: chống kẹt xe ở ngõ cụt xa ({5, 34, 40, 142, 197, 233})
    dead_end_spots = {5, 34, 40, 142, 197, 233}
    dead_end_count = sum(1 for c in patrol_cells if c in dead_end_spots)
    if dead_end_count > 1:
        value -= w.cover_penalty * B * (dead_end_count - 1)

    # V_near: chuỗi có thể với tới trong nửa ngày mai
    if ctx.next_steps > 0:
        reach = 0.5 * ctx.next_steps
        near = 0
        for brand in cfg.brands:
            best = min((_dist_to_brand(ctx, c, brand) for c in patrol_cells), default=INF)
            if best <= reach:
                near += 1
        value += w.near_chain * B * near

    # V_fuel & Fuel Safety:
    # Nếu xe tuần tra kết thúc với fuel < 50 mà xe refuel ở cách xa (> 2 bước), phạt nặng
    for agent in ctx.patrols:
        cell = end_pos[agent]
        fuel = end_fuel[agent]
        if fuel < 50 and car_cells:
            car_dist = min((table.cost(cell, cc) for cc in car_cells), default=INF)
            if car_dist > 2:
                value -= w.strand * B * (1.0 - fuel / 50.0)
        # Điểm thưởng nhiên liệu nhỏ (tiebreaker)
        value += w.fuel_bonus * B * (fuel / max(1, cfg.fuel_limits)) / max(1, ctx.n_patrols)

    # V_road: tự gây tắc cho chính mình
    own_road_stay = sum(v for cell, v in result.stay.items() if cfg.cells[cell] == rules.ROAD)
    value -= w.road_park * B * own_road_stay / max(1, cfg.players)
    return value


def score_day(ctx: DayContext, result: DayResult, truncated: int = 0, dropped_meets: int = 0) -> float:
    """Điểm tổng của một kế hoạch ngày: L trước, rồi D, rồi Q."""
    w = ctx.weights
    B = w.B
    daily_brands = result.brands
    new_brands = len(daily_brands - ctx.visited_brands)
    score = B * len(daily_brands)
    score += ctx.new_brand_value * new_brands
    score += w.serving * result.servings
    score += end_value(ctx, result)
    score -= w.diag_penalty * B * (truncated + dropped_meets)
    return score


def lexicographic_gain(ctx: DayContext, result: DayResult) -> tuple:
    """Phần tăng thật theo thứ tự từ điển, dùng để so sánh và báo cáo."""
    new = len(result.brands - ctx.visited_brands)
    return (new, len(result.brands), result.servings)
