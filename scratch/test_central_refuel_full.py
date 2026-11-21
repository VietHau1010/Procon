import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.strategy import MatchStrategist
from hexudon.planner.roles import centrality

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

strat = MatchStrategist(cfg, search_iterations=500)

# Optimal role assignment based on centrality
c_scores = centrality(cfg)
best_refuel_idx = c_scores.index(min(c_scores))
roles = [rules.PATROL] * cfg.n_agents
roles[best_refuel_idx] = rules.REFUEL
strat.kinds = roles
print(f"Optimal Roles chosen: {roles} (Agent {best_refuel_idx} at cell {cfg.agent_starts[best_refuel_idx]} is Refuel)")

agents = [AgentState(roles[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
road_status = {}

total_servings = 0
total_daily = 0

print("=== 10-DAY SIMULATION WITH CENTRALITY-BASED REFUEL CAR ===")
for day in range(10):
    state = DayState(day=day, agents=agents, road_status=road_status, steps=cfg.day_steps[day])
    decoded = strat.plan_day(state)
    r = decoded.result
    total_servings += r.servings
    total_daily += len(r.brands)
    print(f"Day {day:02d} (steps={cfg.day_steps[day]:3d}): brands={len(r.brands):2d}/{len(cfg.brands)}, servings={r.servings:2d}, ok={r.ok}")
    agents = [AgentState(roles[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]

print("=" * 60)
print(f"TOTAL: L={len(strat.visited_brands)} | D={total_daily} | Q={total_servings}")
print("Team 78 Original: L=19 | D=190 | Q=693")
print("Team 77 Winning:  L=20 | D=200 | Q=737")
