import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.construct import construct
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode
from hexudon.planner.solution import Weights, Meet
import hexudon.planner.alns as alns_mod

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

# 1. Balanced weights: servings are highly valued, fuel bonus does not sacrifice servings
weights = Weights(
    serving=10.0,      # High value on servings
    fuel_bonus=0.02,   # Fuel bonus only as tiebreaker, never sacrifice servings
    end_on_spot=0.2,   # Park on spot for tomorrow
)

# 2. Fix destroy_chain_dup in ALNS
def destroy_chain_dup_safe(ctx, sol, rng, q):
    counts = {}
    for tour in sol.tours:
        for spot in tour:
            counts[spot] = counts.get(spot, 0) + 1
    victims = []
    for slot, tour in enumerate(sol.tours):
        for idx, spot in enumerate(tour):
            if counts[spot] > ctx.stocks[spot]:
                victims.append((slot, idx))
    if not victims:
        return alns_mod.destroy_random(ctx, sol, rng, q)
    rng.shuffle(victims)
    return alns_mod._remove_stops(sol, victims[:q])

alns_mod.DESTROY_FUNCS["chain_dup"] = destroy_chain_dup_safe

kinds = [rules.PATROL, rules.REFUEL, rules.PATROL, rules.PATROL]
agents = [AgentState(kinds[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]

visited_match = set()
total_servings = 0
total_daily_types = 0

print("=== SIMULATING FULL 10 DAYS WITH TUNED ALNS (1000 iters) ===")
for day in range(cfg.n_days):
    ctx = make_context(cfg, day, agents, {}, [s.stocks for s in cfg.spots], visited_match, weights=weights)
    sol = construct(ctx)
    res = alns_mod.improve(ctx, sol, max_iterations=1000)
    dec = res.decoded
    r = dec.result
    visited_match |= r.brands
    total_daily_types += len(r.brands)
    total_servings += r.servings
    print(f"Day {day:02d} (steps={cfg.day_steps[day]:3d}): brands={len(r.brands):2d}/{len(cfg.brands)}, servings={r.servings:2d}, ok={r.ok}, total_L={len(visited_match)}")
    agents = [AgentState(kinds[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]

print("=" * 60)
print(f"FINAL RESULT: L={len(visited_match)}/{len(cfg.brands)} | D={total_daily_types} | Q={total_servings}")
print("Team 78 Original: L=19 | D=190 | Q=693")
print("Team 77 Winning:  L=20 | D=200 | Q=737")
