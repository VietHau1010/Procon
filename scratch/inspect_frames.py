import json

f77_path = "C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json"
r77 = json.load(open(f77_path))

d0 = r77['days'][0]
t77_0 = next(t for t in d0['teams'] if str(t['team_id']) == '77')
t78_0 = next(t for t in d0['teams'] if str(t['team_id']) == '78')

print("T77 frames count:", len(t77_0['frames']))
print("Frame 0 sample:", t77_0['frames'][0])
print("Last frame sample:", t77_0['frames'][-1])
