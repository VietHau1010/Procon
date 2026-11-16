import sys
sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon.planner.engine import make_context
from hexudon.planner.decoder import decode
from hexudon.planner.optimal import OptimalContestPlanner
from hexudon import rules

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg_dict = entry['response']
        if entry.get('path') == '/api/game/state' and entry.get('response', {}).get('day') == 0:
            st0_dict = entry['response']
            break

cfg = MatchConfig.from_json(cfg_dict)
st0_road = {int(k): v for k, v in st0_dict.get('road_condition', {}).items()}
agents = [
    AgentState(rules.PATROL, cfg.agent_starts[0], cfg.fuel_limits),
    AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits),
    AgentState(rules.PATROL, cfg.agent_starts[2], cfg.fuel_limits),
    AgentState(rules.PATROL, cfg.agent_starts[3], cfg.fuel_limits),
]

ctx = make_context(cfg, 0, agents, st0_road, [s.stocks for s in cfg.spots], set())

planner = OptimalContestPlanner(cfg)
routes, plan = planner.plan_day(agents, cfg.day_steps[0], st0_road)
sol = planner.to_day_solution(routes, ctx)
decoded = decode(ctx, sol)

print(f"Decoded from DaySolution: ok={decoded.result.ok}, brands={len(decoded.result.brands)}, servings={decoded.result.servings}, score={decoded.score}")
