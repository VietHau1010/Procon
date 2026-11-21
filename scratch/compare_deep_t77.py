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

with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay_t77 = json.load(f)

strat = MatchStrategist(cfg, search_iterations=500)
roles = strat.select_roles()

agents = [AgentState(roles[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
road_status = {}
spot_positions = {s.pos for s in cfg.spots}

print("=== DETAILED DAY-BY-DAY COMPARISON (OUR PLANNER vs TEAM 77) ===")
for day in range(10):
    d_obj = replay_t77['days'][day]
    t77 = next(t for t in d_obj['teams'] if str(t['team_id']) == '77')
    t77_spots = []
    for aid in [0, 2, 3]:
        cells = []
        for fr in t77['frames']:
            colls = fr.get('collected', [])
            c = fr['agents'][aid]['cell']
            if c in colls and (not cells or cells[-1] != c):
                cells.append(c)
        t77_spots.append(cells)
    
    t77_end_pos = [t77['frames'][-1]['agents'][i]['cell'] for i in range(4)]
    t77_end_fuel = [t77['frames'][-1]['agents'][i]['fuel'] for i in range(4)]
    
    # Run our planner
    state = DayState(day=day, agents=agents, road_status=road_status, steps=cfg.day_steps[day])
    decoded = strat.plan_day(state)
    r = decoded.result
    
    patrol_agents = [i for i, k in enumerate(roles) if k == rules.PATROL]
    
    print(f"\n--- DAY {day} (steps={cfg.day_steps[day]}) ---")
    print(f"Team 77: serv={t77['servings']}, end_pos={t77_end_pos}, end_fuel={t77_end_fuel}")
    for p_idx in range(3):
        print(f"  T77 P{p_idx} ({len(t77_spots[p_idx])} spots): {t77_spots[p_idx]}")
    
    # Check our decoded solution
    # Let's inspect last_sent or decoded solution
    if hasattr(strat, 'last_sent') and strat.last_sent:
        sol = strat.last_sent
    print(f"Our Bot: serv={r.servings}, end_pos={r.end_pos}, end_fuel={r.end_fuel}")
    
    agents = [AgentState(roles[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]
