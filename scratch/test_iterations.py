import sys
sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.strategy import MatchStrategist as ContestStrategist

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg_json = obj['response']
            break

cfg = MatchConfig.from_json(cfg_json)

for iters in [50, 200, 500, 1000]:
    strat = ContestStrategist(cfg, search_iterations=iters)
    kinds = strat.select_roles()
    agents = [AgentState(kinds[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
    road_status = {}
    total_servings = 0
    total_daily_brands = 0

    for day in range(cfg.n_days):
        d_state = DayState(day=day, agents=agents, road_status=road_status)
        decoded = strat.plan_day(d_state)
        res = decoded.result
        total_servings += res.servings
        total_daily_brands += len(res.brands)
        agents = [AgentState(kinds[i], res.end_pos[i], res.end_fuel[i]) for i in range(cfg.n_agents)]

    print(f"Iters: {iters:4d} | L: {len(strat.visited_brands):2d}/{len(cfg.brands)} | D: {total_daily_brands:3d} | Q: {total_servings:3d}")
