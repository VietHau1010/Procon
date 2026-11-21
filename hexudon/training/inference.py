"""Explicit opt-in use of a learned policy; never patch live planner defaults."""

from .storage import load_baseline, load_policy


def trained_strategist(directory, config, model_path=None, **kwargs):
    """Build the frozen strategist with learned weights and an input MatchConfig.

    Test with a real wall-clock budget before using this in a contest client.
    Offline promotion evaluates L/D/Q, not response-time ranking or network I/O.
    """
    weights = load_policy(directory, model_path)
    api = load_baseline(directory)
    obj = config.to_json() if hasattr(config, "to_json") else config
    frozen_config = api["model"].MatchConfig.from_json(obj)
    if "weights" in kwargs:
        raise ValueError("weights phải lấy từ model đã lưu, không truyền trực tiếp")
    return api["strategy"].MatchStrategist(
        frozen_config, weights=api["solution"].Weights(**weights), **kwargs)
