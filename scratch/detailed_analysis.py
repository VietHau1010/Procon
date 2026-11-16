import json

f77_path = "C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json"
f78_path = "replays/contest-087d9b9a-t78-20261112-165702-replay.json"

r77 = json.load(open(f77_path))
r78 = json.load(open(f78_path))

d0 = r77['days'][0]
print("Keys of team 0:", list(d0['teams'][0].keys()))
for t in d0['teams']:
    tid = t.get('team_id') or t.get('id')
    if tid in (77, 78, "77", "78"):
        print(f"Team {tid}: keys={list(t.keys())}")
        if 'score' in t:
            print(f"  score={t['score']}")
        if 'agents' in t:
            print(f"  agents count={len(t['agents'])}")
            for a_idx, a in enumerate(t['agents']):
                print(f"    agent {a_idx}: keys={list(a.keys())}")
                if 'history' in a:
                    print(f"      history len={len(a['history'])}")
                if 'trajectory' in a:
                    print(f"      trajectory len={len(a['trajectory'])}")
                if 'path' in a:
                    print(f"      path len={len(a['path'])}")
