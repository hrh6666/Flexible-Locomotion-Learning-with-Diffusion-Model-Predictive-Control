
from legged_gym.envs.base.legged_robot_field import LeggedRobotField
from legged_gym.envs.base.legged_robot_mix import LeggedRobotMix
from legged_gym.envs.base.legged_robot_noisy import LeggedRobotNoisyMixin
from legged_gym.envs.base.legged_robot_threemix import LeggedRobotThreeMix

class RobotFieldNoisy(LeggedRobotNoisyMixin, LeggedRobotField):
    """ Using inheritance to combine the two classes.
    Then, LeggedRobotNoisyMixin and LeggedRobot can also be used elsewhere.
    """
    pass

class RobotFieldMixNoisy(LeggedRobotNoisyMixin, LeggedRobotMix):
    """ Using inheritance to combine the two classes.
    Then, LeggedRobotNoisyMixin and LeggedRobot can also be used elsewhere.
    """
    pass

class RobotFieldThreeMixNoisy(LeggedRobotNoisyMixin, LeggedRobotThreeMix):
    """ Using inheritance to combine the two classes.
    Then, LeggedRobotNoisyMixin and LeggedRobot can also be used elsewhere.
    """
    pass
