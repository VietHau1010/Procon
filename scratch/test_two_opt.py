import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.construct import construct, InsertionState, add_fuel_meets
from hexudon.planner.decoder import decode

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

agents = [AgentState(rules.PATROL, cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
agents[1] = AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits)
ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())

def two_opt_tour(ctx, slot, tour):
    """Untangle tour using 2-opt swaps."""
    if len(tour) < 3:
        return tour
    cells = [ctx.start_of_patrol(slot)] + [ctx.spot_cells[s] for s in tour]
    best_tour = list(tour)
    table = ctx.dist_step
    
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_tour)):
            for j in range(i + 1, len(best_tour) + 1):
                # evaluate reversing segment [i-1 .. j-1]
                # current: (i-1 -> i) and (j-1 -> j if j < len else 0)
                # new: (i-1 -> j-1) and (i -> j if j < len else 0)
                c_prev = cells[i - 1]
                c_i = cells[i]
                c_j_minus_1 = cells[j]
                c_j = cells[j + 1] if j + 1 < len(cells) else None
                
                cur_dist = table.cost(c_prev, c_i) + (table.cost(c_j_minus_1, c_j) if c_j is not None else 0)
                new_dist = table.cost(c_prev, c_j_minus_1) + (table.cost(c_i, c_j) if c_j is not None else 0)
                
                if new_dist < cur_dist:
                    # Reverse in tour
                    best_tour[i-1:j] = reversed(best_tour[i-1:j])
                    cells = [ctx.start_of_patrol(slot)] + [ctx.spot_cells[s] for s in best_tour]
                    improved = True
                    break
            if improved:
                break
    return best_tour

sol = construct(ctx)
first = decode(ctx, sol)
print("Before 2-Opt:", first.result.servings, "servings, brands:", len(first.result.brands))

# Apply 2-opt to all tours
for slot in range(ctx.n_patrols):
    sol.tours[slot] = two_opt_tour(ctx, slot, sol.tours[slot])

# Now refill with greedy fill
state = InsertionState(ctx, sol)
for _ in range(100):
    best = state.best_insertion()
    if best is None:
        break
    state.apply(*best)

sol_opt = add_fuel_meets(ctx, state.sol)
after = decode(ctx, sol_opt)
print("After 2-Opt + Greedy Refill:", after.result.servings, "servings, brands:", len(after.result.brands))
