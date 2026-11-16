import sys; sys.path.insert(0, '.')
import json

with open("C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json") as f:
    replay_t77 = json.load(f)

for day in range(10):
    d_obj = replay_t77['days'][day]
    t77 = next(t for t in d_obj['teams'] if str(t['team_id']) == '77')
    spots = [[], [], []]
    for aid_idx, aid in enumerate([0, 2, 3]):
        for fr in t77['frames']:
            colls = fr.get('collected', [])
            c = fr['agents'][aid]['cell']
            if c in colls and (not spots[aid_idx] or spots[aid_idx][-1] != c):
                spots[aid_idx].append(c)
    print(f"=== DAY {day} (steps={d_obj['steps']}) ===")
    for aid_idx in range(3):
        print(f"  P{aid_idx} ({len(spots[aid_idx])} spots): {spots[aid_idx]}")
