"""Map-independent features of a candidate's end-of-day state.

No feature reads a rollout label, a future opponent action or a cell ID.
"""

import math
from collections import Counter

FEATURES = (
    "days_left_fraction", "next_step_fraction", "remaining_step_fraction",
    "tank_size_vs_day", "patrol_fraction", "brand_fraction", "spot_fraction",
    "unseen_brand_fraction", "fuel_mean", "fuel_min", "fuel_std", "stranded_fraction",
    "co_located_tanker_fraction", "tanker_distance_mean", "tanker_distance_max",
    "patrol_spread", "parked_spot_fraction", "free_brand_fraction", "free_serving_fraction",
    "reachable_brand_fraction", "reachable_unseen_fraction", "brand_distance_mean",
    "fuel_reachable_brand_fraction", "own_road_stay_fraction", "next_busy_fraction",
    "next_jammed_fraction", "patrol_road_fraction",
)


def _mean(values):
    return sum(values) / max(1, len(values))


def _relative(value, scale):
    if not math.isfinite(value):
        return 2.0
    return min(2.0, max(0.0, value / max(1, scale)))


def future_bounds(ctx, result):
    cfg = ctx.config
    if ctx.days_left <= 0:
        return [0, 0, 0]
    unseen = len(set(cfg.brands) - ctx.visited_brands - result.brands)
    q_day = sum(min(s.stocks, ctx.n_patrols) for s in cfg.spots)
    return [unseen, ctx.days_left * len(cfg.brands), ctx.days_left * q_day]


def extract(ctx, result, modules, forecast, yesterday_status=None, yesterday_stay=None):
    """Use only configuration, history and the candidate's simulated result."""
    if not result.ok:
        raise ValueError("Không trích đặc trưng từ kế hoạch không hợp lệ")
    cfg = ctx.config
    patrols = [result.end_pos[i] for i in ctx.patrols]
    tankers = [result.end_pos[i] for i in ctx.cars]
    fuel = [result.end_fuel[i] for i in ctx.patrols]
    fuel_ratio = [f / max(1, cfg.fuel_limits) for f in fuel]
    fuel_mean = _mean(fuel_ratio)
    next_steps = cfg.steps_of_day(ctx.day + 1) if ctx.days_left else 0
    unseen = set(cfg.brands) - ctx.visited_brands - result.brands
    status = forecast.predict(ctx.road_status, yesterday_status,
                              own_stay_today=result.stay,
                              own_stay_yesterday=yesterday_stay, floor_today=ctx.day == 1)
    graph = modules["pathing"].DayGraph(cfg, status)
    sources = sorted(set(patrols + tankers))
    steps = modules["pathing"].DistTable(graph, sources, "step")
    eco = modules["pathing"].DistTable(graph, sources, "fuel")
    by_brand = {}
    fuel_access = {}
    for brand in cfg.brands:
        cells = [cfg.spots[si].pos for si in cfg.spots_by_brand[brand] if cfg.spots[si].stocks > 0]
        by_brand[brand] = min((steps.cost(p, cell) for p in patrols for cell in cells), default=float("inf"))
        fuel_access[brand] = any(
            eco.cost(p, cell) <= f and eco.alt_cost(p, cell) <= next_steps
            for p, f in zip(patrols, fuel) for cell in cells)
    tanker_dist = [min((steps.cost(car, p) for car in tankers), default=float("inf")) for p in patrols]
    spread = [steps.cost(a, b) for i, a in enumerate(patrols) for b in patrols[i+1:]]
    parked = Counter(p for p in patrols if p in cfg.spot_index_at)
    free_brands = {cfg.spots[cfg.spot_index_at[p]].brand for p in parked
                   if cfg.spots[cfg.spot_index_at[p]].stocks > 0}
    free_q = sum(min(count, cfg.spots[cfg.spot_index_at[p]].stocks) for p, count in parked.items())
    q_day = sum(min(s.stocks, len(patrols)) for s in cfg.spots)
    road = cfg.road_cells
    values = [
        ctx.days_left / max(1, cfg.n_days),
        next_steps / max(1, max(cfg.day_steps)),
        sum(cfg.day_steps[ctx.day+1:]) / max(1, sum(cfg.day_steps)),
        _relative(cfg.fuel_limits, max(cfg.day_steps)),
        len(patrols) / max(1, cfg.n_agents), len(cfg.brands) / 32.0, len(cfg.spots) / 32.0,
        len(unseen) / max(1, len(cfg.brands)),
        fuel_mean, min(fuel_ratio, default=0.0),
        math.sqrt(_mean([(f-fuel_mean)**2 for f in fuel_ratio])),
        sum(f < modules["rules"].fuel_cost(cfg.cells[p]) for p, f in zip(patrols, fuel)) / max(1, len(patrols)),
        sum(p in tankers for p in patrols) / max(1, len(patrols)),
        _mean([_relative(d, next_steps) for d in tanker_dist]),
        max((_relative(d, next_steps) for d in tanker_dist), default=2.0),
        _mean([_relative(d, next_steps) for d in spread]),
        sum(p in cfg.spot_index_at for p in patrols) / max(1, len(patrols)),
        len(free_brands) / max(1, len(cfg.brands)), free_q / max(1, q_day),
        sum(d <= next_steps for d in by_brand.values()) / max(1, len(cfg.brands)),
        sum(by_brand[b] <= next_steps for b in unseen) / max(1, len(unseen)),
        _mean([_relative(d, next_steps) for d in by_brand.values()]),
        sum(fuel_access.values()) / max(1, len(cfg.brands)),
        sum(result.stay.get(c, 0) for c in road) / max(1, ctx.steps * cfg.n_agents),
        sum(status.get(c, 0) >= 1 for c in road) / max(1, len(road)),
        sum(status.get(c, 0) >= 2 for c in road) / max(1, len(road)),
        sum(cfg.cells[p] == 1 for p in patrols) / max(1, len(patrols)),
    ]
    if len(values) != len(FEATURES) or not all(math.isfinite(x) for x in values):
        raise ValueError("Feature contract không hợp lệ")
    return values
