"""Contest planner: ALNS, coverage rescue, stock-rich beam search and Q fill.

MatchStrategist/plan_day enable the stock-rich strategy by default. Frozen
training-run models remain separate and are never rewritten by importing this
package.
"""

from .alns import improve
from .construct import construct, greedy_fill, repair
from .decoder import Decoded, decode
from .coverage import rescue_coverage, rescue_servings
from .dense_search import abundant, beam_tours, improve_dense
from .engine import make_context, plan_day, safe_plan, safe_solution
from .objective import default_contest_weights, end_value, lexicographic_gain, score_day
from .roles import choose_roles_for_match, select_roles
from .strategy import MatchStrategist

__all__ = [
    "construct",
    "greedy_fill",
    "repair",
    "decode",
    "Decoded",
    "improve",
    "end_value",
    "lexicographic_gain",
    "score_day",
    "default_contest_weights",
    "make_context",
    "plan_day",
    "safe_plan",
    "safe_solution",
    "choose_roles_for_match",
    "select_roles",
    "MatchStrategist",
    "rescue_coverage",
    "rescue_servings",
    "abundant",
    "beam_tours",
    "improve_dense",
]
