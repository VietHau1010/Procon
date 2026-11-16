import sys
sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon.rules import PATROL, REFUEL
from hexudon.pathing import DayGraph, DistTable, LegBook
from hexudon.planner.solution import build_context, Weights
from hexudon.planner.construct import construct, InsertionState, SpotNeighbours

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg_dict = entry['response']
        if entry.get('path') == '/api/game/state' and entry.get('response', {}).get('day') == 0:
            st0_dict = entry['response']
            break

cfg = MatchConfig.from_json(cfg_dict)
road_status = {int(k): v for k, v in st0_dict.get('road_condition', {}).items()}
graph = DayGraph(cfg, road_status)

spots = cfg.spots
sources = list(cfg.agent_starts) + [s.pos for s in spots]
dt_step = DistTable(graph, sources, metric="step")
dt_fuel = DistTable(graph, sources, metric="fuel")
book = LegBook(cfg, dt_step, dt_fuel)

agents = [
    AgentState(PATROL, cfg.agent_starts[0], cfg.fuel_limits),
    AgentState(REFUEL, cfg.agent_starts[1], cfg.fuel_limits),
    AgentState(PATROL, cfg.agent_starts[2], cfg.fuel_limits),
    AgentState(PATROL, cfg.agent_starts[3], cfg.fuel_limits),
]
stocks = [s.stocks for s in cfg.spots]

ctx = build_context(
    config=cfg,
    day=0,
    agents=agents,
    road_status=road_status,
    stocks=stocks,
    visited_brands=set(),
    book=book,
    dist_step=dt_step,
)

# Test 1: Original construct (NEAR_K = 10)
sol1 = construct(ctx)
brands1 = {ctx.spot_brand[sp] for t in sol1.tours for sp in t}
print(f"Original construct (NEAR_K=10): {len(brands1)} brands. Missing: {set(range(20)) - brands1}")
print(f"Meets: {sol1.meets}")

# Test 2: Modify SpotNeighbours to use k = 30
def custom_init(self, ctx, k=30):
    self.ctx = ctx
    self.k = k
    self._cache = {}

SpotNeighbours.__init__ = custom_init
sol2 = construct(ctx)
brands2 = {ctx.spot_brand[sp] for t in sol2.tours for sp in t}
print(f"Modified construct (k=30): {len(brands2)} brands. Missing: {set(range(20)) - brands2}")
print(f"Meets: {sol2.meets}")
for i, t in enumerate(sol2.tours):
    cells = [ctx.spot_cells[s] for s in t]
    print(f"  Tour {i} cells: {cells} (808 in tour? {808 in cells})")
