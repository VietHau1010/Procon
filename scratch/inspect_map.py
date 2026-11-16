import json

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line_idx, line in enumerate(f):
        entry = json.loads(line)
        path = entry.get('path', '')
        if 'match' in path or line_idx < 5:
            print(f"Line {line_idx}: {entry.get('method')} {path} -> {list(entry.get('response', {}).keys()) if isinstance(entry.get('response'), dict) else type(entry.get('response'))}")
            if 'map' in entry.get('response', {}):
                print("Found 'map' in response!")
                break
