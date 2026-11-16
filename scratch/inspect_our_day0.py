import sys
sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.construct import construct
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

sol = construct(ctx)
dec = decode(ctx, sol)
print("Construct score:", dec.score, "servings:", dec.result.servings, "brands:", len(dec.result.brands))
print("Tours:", sol.tours)
print("Meets:", sol.meets)
print("Waits:", sol.waits)
for slot in range(ctx.n_patrols):
    tour_spots = sol.tours[slot]
    tour_cells = [ctx.spot_cells[s] for s in tour_spots]
    print(f"Patrol {slot}: len={len(tour_spots)}, cells={tour_cells}")
print("End positions:", dec.result.end_pos)
print("End fuels:    ", dec.result.end_fuel)
