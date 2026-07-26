"""Task planner interface. / 태스크 플래너 인터페이스.

A task planner turns the symbolic state of a scene into a sequence of moves. It knows nothing
about geometry; the skill layer resolves each move into object and goal positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Move:
    """Move ``item`` from ``source`` to ``destination``. / 물체 하나의 이동."""

    item: str
    source: str
    destination: str


class TaskPlanner(Protocol):
    """Produces the move sequence for a task state. / 태스크 상태 → 이동 순서."""

    def plan(self, state: dict[str, list[str]], goal: str) -> list[Move]: ...
