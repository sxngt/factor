"""Tower of Hanoi task plugin. / 하노이 탑 태스크 플러그인.

The tower is three pegs in a row on a base plate; rings of decreasing diameter are stacked and
moved one at a time under the usual rule (a ring may only rest on a larger ring). The plugin
supplies the geometry (peg layout, ring heights) and the shortest-move planner.
"""

from factor.tasks.hanoi.solver import HanoiPlanner, solve_hanoi
from factor.tasks.hanoi.spec import HanoiLayout, Ring

__all__ = ["HanoiLayout", "HanoiPlanner", "Ring", "solve_hanoi"]
