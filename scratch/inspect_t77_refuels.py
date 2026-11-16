import sys; sys.path.insert(0, '.')
import json

with open("C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json") as f:
    replay_t77 = json.load(f)

for day in range(10):
    d_obj = replay_t77['days'][day]
    t77 = next(t for t in d_obj['teams'] if str(t['team_id']) == '77')
    refuels = []
    for aid in [0, 2, 3]:
        for fr_idx in range(1, len(t77['frames'])):
            f_prev = t77['frames'][fr_idx-1]['agents'][aid]['fuel']
            f_cur = t77['frames'][fr_idx]['agents'][aid]['fuel']
            if f_cur > f_prev:
                step = t77['frames'][fr_idx]['step']
                cell = t77['frames'][fr_idx]['agents'][aid]['cell']
                refuels.append((aid, step, cell, f_prev, f_cur))
    print(f"Day {day} refuels: {refuels}")
