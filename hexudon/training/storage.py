"""Versioned JSON artifacts, source snapshots, checksums and single-writer locks."""

import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from contextlib import contextmanager

SCHEMA = 1
# Do not train the rules or the score priority. Only strategy heuristics vary.
SPACE = {
    "serving": (5.0, 40.0),
    "urgency": (0.05, 2.0),
    "end_on_spot": (0.01, 1.0),
    "near_chain": (0.01, 1.0),
    "strand": (0.1, 5.0),
    "fuel_bonus": (0.001, 0.5),
    "road_park": (0.001, 0.3),
    "diag_penalty": (0.001, 0.1),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def writer_lock(directory):
    path = Path(directory) / ".writer.lock"
    try:
        fh = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError("Run đang bị khóa (.writer.lock). Không chạy hai trainer cùng run. "
                         "Nếu tiến trình bị kill, kiểm tra đã dừng rồi mới xóa lock.") from exc
    try:
        fh.write(str(os.getpid()))
        fh.close()
        yield
    finally:
        fh.close()
        path.unlink(missing_ok=True)


def snapshot_sources(directory):
    """Copy executable baseline dependencies, never .env, clients or replays."""
    source = Path(__file__).resolve().parents[1]
    files = [source / name for name in (
        "__init__.py", "model.py", "geometry.py", "rules.py", "simulator.py",
        "traffic.py", "pathing.py")]
    for name in ("planner", "planner_contest"):
        files.extend(sorted((source / name).glob("*.py")))
    hashes = {}
    for path in files:
        relative = path.relative_to(source).as_posix()
        data = path.read_bytes()
        target = Path(directory) / "baseline" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = digest(data)
    return hashes


def verify_run(directory):
    root = Path(directory)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Training schema không được hỗ trợ")
    identity = dict(manifest)
    claimed = identity.pop("baseline_id", None)
    if digest(json_bytes(identity)) != claimed:
        raise ValueError("Manifest đã thay đổi; tạo run mới thay vì thay baseline")
    for name, expected in manifest.get("runtime", {}).items():
        if digest((Path(__file__).parent / name).read_bytes()) != expected:
            raise ValueError(f"Training runtime đã thay đổi: {name}; tạo run mới để so sánh công bằng")
    for relative, expected in manifest["sources"].items():
        path = root / "baseline" / relative
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"Baseline source bị thay đổi: {relative}")
    for split, expected in manifest["datasets"].items():
        path = root / "data" / (split + ".json")
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Dataset bị thay đổi: {split}")
    return manifest


def load_baseline(directory):
    """Load a separate package so all relative imports refer to frozen code."""
    root = Path(directory).resolve() / "baseline"
    # Include directory: two runs with identical sources may have different maps.
    name = "_hexudon_training_" + digest(str(root).encode())[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return {
        "model": importlib.import_module(name + ".model"),
        "simulator": importlib.import_module(name + ".simulator"),
        "strategy": importlib.import_module(name + ".planner_contest.strategy"),
        "objective": importlib.import_module(name + ".planner_contest.objective"),
        "solution": importlib.import_module(name + ".planner.solution"),
    }


def validate_weights(weights, baseline):
    if set(weights) != set(baseline):
        raise ValueError("Model thiếu/thừa trọng số")
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Trọng số không hữu hạn: {key}")
        if key in SPACE:
            low, high = SPACE[key]
            if not low <= value <= high:
                raise ValueError(f"Trọng số ngoài giới hạn: {key}")
        elif value != baseline[key]:
            raise ValueError(f"Không được học/thay trọng số khóa: {key}")
    return dict(weights)


def policy_artifact(manifest, weights, trial):
    return {"schema": SCHEMA, "kind": "hexudon-evolutionary-weights",
            "baseline_id": manifest["baseline_id"], "trial": trial,
            "weights": dict(weights), "automatic_live_deployment": False}


def load_policy(directory, path=None):
    manifest = verify_run(directory)
    artifact = read_json(path or Path(directory) / "champion.json")
    if (artifact.get("schema") != SCHEMA
            or artifact.get("kind") != "hexudon-evolutionary-weights"
            or artifact.get("baseline_id") != manifest["baseline_id"]):
        raise ValueError("Model không thuộc baseline của run này")
    return validate_weights(artifact["weights"], manifest["baseline_weights"])
