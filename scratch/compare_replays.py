import json

with open("C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json") as f:
    replay = json.load(f)

print("%-4s | %-5s | %-8s | %-9s | %-8s | %-9s" % ("Day", "Steps", "T77 Serv", "T77 Types", "T78 Serv", "T78 Types"))
print("-" * 55)

tot_s77, tot_s78 = 0, 0
for d_idx, day_obj in enumerate(replay["days"]):
    steps = day_obj["steps"]
    teams_dict = {str(t["team_id"]): t for t in day_obj["teams"]}
    t77 = teams_dict.get("77", {})
    t78 = teams_dict.get("78", {})
    s77 = t77.get("servings", 0)
    s78 = t78.get("servings", 0)
    typ77 = t77.get("types", 0)
    typ78 = t78.get("types", 0)
    tot_s77 += s77
    tot_s78 += s78
    print("%-4d | %-5d | %-8d | %-9d | %-8d | %-9d" % (d_idx, steps, s77, typ77, s78, typ78))

print("-" * 55)
print("Total:     | %-8d |           | %-8d |" % (tot_s77, tot_s78))
