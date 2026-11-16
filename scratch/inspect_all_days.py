import json

f77_path = "C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json"
f78_path = "replays/contest-087d9b9a-t78-20261112-165702-replay.json"

r77 = json.load(open(f77_path))
r78 = json.load(open(f78_path))

print("=== DAILY RENDEZVOUS & TANKER ANALYSIS ===")
for d_idx in range(10):
    d77 = r77['days'][d_idx]
    d78 = r78['days'][d_idx]
    
    t77 = next(t for t in d77['teams'] if str(t['team_id']) == '77')
    t78 = next(t for t in d78['teams'] if str(t['team_id']) == '78')
    
    # T77 refuel events
    refuels_77 = []
    for fr_idx in range(1, len(t77['frames'])):
        p = t77['frames'][fr_idx - 1]
        c = t77['frames'][fr_idx]
        for a in range(len(c['agents'])):
            if c['agents'][a]['fuel'] > p['agents'][a]['fuel']:
                refuels_77.append((c['step'], a, c['agents'][a]['cell']))
                
    # T78 refuel events
    refuels_78 = []
    for fr_idx in range(1, len(t78['frames'])):
        p = t78['frames'][fr_idx - 1]
        c = t78['frames'][fr_idx]
        for a in range(len(c['agents'])):
            if c['agents'][a]['fuel'] > p['agents'][a]['fuel']:
                refuels_78.append((c['step'], a, c['agents'][a]['cell']))
                
    # End of day positions
    end_77 = [(a['cell'], a['fuel']) for a in t77['frames'][-1]['agents']]
    end_78 = [(a['cell'], a['fuel']) for a in t78['frames'][-1]['agents']]
    
    print(f"\nDay {d_idx} (steps={len(t77['frames'])}):")
    print(f"  T77 refuels: {refuels_77}")
    print(f"  T77 end pos: {end_77}")
    print(f"  T78 refuels: {refuels_78}")
    print(f"  T78 end pos: {end_78}")
