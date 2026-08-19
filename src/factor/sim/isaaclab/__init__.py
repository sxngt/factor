"""Isaac Lab implementation of the simulation twin. / Isaac Lab 트윈 구현.

Isaac Lab modules can only be imported after the simulation application has been launched, so
this package resolves its public names lazily::

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    from factor.sim.isaaclab import HanoiTwin, make_env_cfg, settings_from_scene
"""

from __future__ import annotations

__all__ = ["HanoiTwin", "HanoiTwinCfg", "TwinSettings", "make_env_cfg", "settings_from_scene"]


def __getattr__(name: str):
    if name in __all__:
        from factor.sim.isaaclab import twin

        return getattr(twin, name)
    raise AttributeError(name)
