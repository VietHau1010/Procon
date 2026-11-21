"""Optimistic daily bounds, not a claim that a joint route is feasible."""

from ..planner.solution import DayContext


def day_bounds(ctx: DayContext) -> dict:
    """Relax fuel and competition for route time; retain reachability and stock.

    A patrol collects at most once per spot. Individual shortest travel times
    are lower bounds, so counting each independently gives an upper bound even
    when those visits cannot all fit on the same route.
    """
    eligible = []
    brands = set()
    for si, cell in enumerate(ctx.spot_cells):
        count = sum(ctx.dist_step.cost(ctx.start_of_patrol(p), cell) <= ctx.steps
                    for p in range(ctx.n_patrols))
        count = min(max(0, ctx.stocks[si]), count)
        eligible.append(count)
        if count:
            brands.add(ctx.spot_brand[si])
    return {"types": len(brands), "servings": sum(eligible),
            "new_types": len(brands - ctx.visited_brands),
            "relaxation": "independent visits; fuel and joint travel budget relaxed"}
