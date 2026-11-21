"""Điểm tìm kiếm: lợi ích hôm nay cộng ước lượng giá trị cuối ngày.

Đây là heuristic cho cả trận, không phải thứ tự từ điển chính xác của một
ngày: có thể đổi điểm hôm nay lấy vị trí/xăng cho ngày mai. Kết quả benchmark
và chọn vai trò được so bằng tuple (L, D, Q) thật. Điểm đỗ xe được giới hạn
theo số nhãn và kho, không phụ thuộc ID ô của một map cụ thể.
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
        # Experimental learned weight, enabled by user request. Q03 runs
        # observed 763 and 819, but differed in tanker and wall-clock search;
        # they do not establish a causal gain or a multi-map improvement.
        road_park=0.0387433,
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

    # Value tomorrow's free collections once per brand and within its stock.
    spot_at = cfg.spot_index_at
    parked = {}
    for c in patrol_cells:
        if c in spot_at:
            parked[c] = parked.get(c, 0) + 1
    free_brands = set()
    for cell, count in parked.items():
        spot = cfg.spots[spot_at[cell]]
        servings = min(count, spot.stocks)
        if servings > 0:
            free_brands.add(spot.brand)
            value += w.serving * servings
    value += w.end_on_spot * B * len(free_brands)

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


def coverage_key(ctx: DayContext, decoded) -> tuple:
    """Archive actual collected types before heuristic position/fuel/Q value.

    Keep the scalar search objective for exploration and terminal valuation;
    only selection of the retained plan gives L/D strict priority. Counting
    planned stops here would incorrectly reward routes truncated by decoding.
    """
    if not decoded.result.ok:
        return (-1, -1, float("-inf"))
    new, daily, servings = lexicographic_gain(ctx, decoded.result)
    return (new, daily, servings, decoded.score)
