"""AgileX Hunter SE robot model -- a thin, config-driven RobotModel
implementation. All numbers live in config/robot/hunter_se.yaml; nothing
here is hardcoded (section 45)."""

from __future__ import annotations

from hunter_kinodynamic_rl.config.schema import RobotConfig
from hunter_kinodynamic_rl.robot.limits import RobotLimits


class HunterSE(RobotLimits):
    """Concrete :class:`~hunter_kinodynamic_rl.robot.interface.RobotModel`
    for the AgileX Hunter SE. Swapping robots means adding a sibling class
    (e.g. ``Scout(RobotLimits)``) plus its own ``config/robot/scout.yaml`` --
    every other module only depends on the ``RobotModel`` Protocol."""

    def __init__(self, config: RobotConfig):
        if config.name != "hunter_se":
            raise ValueError(f"HunterSE requires robot.name == 'hunter_se', got {config.name!r}")
        super().__init__(config)
