"""Skills: closed-loop pick / place behaviours. / 스킬 층: 폐루프 집기·놓기.

A skill composes the funnel planner, the contact search and the perception corrections into a
per-tick command. Rule-based skills are the initialisation for reinforcement-learning
fine-tuning in a contact simulation.
"""

from factor.skills.pick_place import PickPlaceSkill, SkillCommand, SkillObservation

__all__ = ["PickPlaceSkill", "SkillCommand", "SkillObservation"]
