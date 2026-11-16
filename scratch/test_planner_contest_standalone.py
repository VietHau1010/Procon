import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.strategy import MatchStrategist

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

strat = MatchStrategist(cfg, search_iterations=500)
roles = strat.select_roles()
print(f"Roles chosen: {roles}")

agents = [AgentState(roles[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
road_status = {}

print("=== RUNNING 10-DAY CONTEST SIMULATION VIA MATCHSTRATEGIST ===")
for day in range(10):
    state = DayState(
        day=day,
        agents=agents,
        road_status=road_status,
        steps=cfg.day_steps[day],
    )
    decoded = strat.plan_day(state)
    r = decoded.result
    print(f"Day {day:02d} (steps={cfg.day_steps[day]:3d}): brands={len(r.brands):2d}/{len(cfg.brands)}, servings={r.servings:2d}, ok={r.ok}, total_L={len(strat.visited_brands)}, total_D={strat.daily_types_total}, total_Q={strat.servings_total}")
    agents = [AgentState(roles[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]

print("=" * 60)
print(f"FINAL MATCH RESULT:")
print(f"  L (Total Brands) = {len(strat.visited_brands)} / {len(cfg.brands)}")
print(f"  D (Daily Brands) = {strat.daily_types_total}")
print(f"  Q (Total Servings) = {strat.servings_total}")
print("=" * 60)
print(f"Baseline Team 78: L=19 | D=190 | Q=693")
print(f"Baseline Team 77: L=20 | D=200 | Q=737")
