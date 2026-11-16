import json
from hexudon.model import MapState, Spot, AgentState
from hexudon.planner.strategy import StrategyPlanner
from hexudon.planner.solution import PlanRequest
from hexudon.pathing import HexGrid

# Load config and day 0 state from jsonl
with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg = entry['response']
        if entry.get('path') == '/api/game/state' and entry.get('response', {}).get('day') == 0:
            st0 = entry['response']
            break

print("Config loaded. Checking why cell 808 was not in T78 plan on Day 0...")
# Let's inspect the request for day 0
from hexudon.rules import GameConfig
# Let's see how the client calls strategy
