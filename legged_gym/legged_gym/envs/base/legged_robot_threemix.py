from collections import OrderedDict, defaultdict
import itertools
from threading import Condition
import numpy as np
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, quat_apply_yaw_inverse, quat_apply
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import get_terrain_cls
from legged_gym.utils.observation import get_obs_slice
from .legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.legged_robot_field import LeggedRobotField
from legged_gym.envs.base.legged_robot_noisy import LeggedRobotNoisyMixin
from legged_gym.envs.base.legged_robot_mix import LeggedRobotMix

class LeggedRobotThreeMix(LeggedRobotMix):
    """ Using inheritance to ensure clean mixture of quadrupedal, bipedal, and three-legged environments. """
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        print("Using LeggedRobotThreeMix.__init__, num_obs and num_privileged_obs will be computed instead of assigned.")
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Default is_three_foot state (assuming no legs are touching the ground)
        is_three_foot_default = torch.zeros_like(self.is_bipedal, dtype=torch.bool)
        
        # Update is_three_foot based on terrain columns
        self.is_three_foot = is_three_foot_default
        
    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()

        super()._post_physics_step_callback()  # Call the base class method
        
        # Check if we are in a three-foot terrain area (left front leg does not touch the ground)
        col_indices = torch.floor(self.root_states[:, 1] / self.cfg.terrain.terrain_width).long()
        is_three_foot_new = torch.zeros_like(self.is_bipedal, dtype=torch.bool)  # Default for three-foot
        
        # Check for three-foot column-based logic
        for col in self.cfg.terrain.three_foot_cols:  # Add support for three-foot columns
            mask = col_indices == col
            # Assuming left front leg does not touch ground when Z position is less than terrain height * 3
            is_three_foot_new = torch.where(mask, 
                                            self.root_states[:, 2] <= self.cfg.terrain.terrain_width * 3, 
                                            is_three_foot_new)

        self.is_three_foot = is_three_foot_new  # Update three-foot status

    def check_termination(self):
        """ Check if environments need to be reset """
        super().check_termination()  # Call the base class method

        # Add the termination condition for three-foot robots
        if "three_foot_contact" in self.cfg.termination.termination_terms:
            body_contact = (torch.norm(self.contact_forces[:, self.terminate_three_foot_contact_indices, :], dim=-1) > 1.0)  # Shape: (N, num_collision_bodies)

            # Determine if any monitored body exceeds the contact force threshold
            force_exceed_criteria = torch.any(body_contact, dim=1)
            self.reset_buf |= force_exceed_criteria * self.is_three_foot

    def _get_one_hot_obs(self, privileged=False):
        """ Return one-hot observation with an additional dimension for three-legged robots. """
        # [1, 0, 0] for quadrupedal, [0, 1, 0] for three-legged, [0, 0, 1] for bipedal
        obs_buf = torch.zeros((self.num_envs, 3), dtype=torch.float32).to(self.device)
        
        # Set the appropriate one-hot encoding based on the robot's state
        obs_buf[:, 0] = self.is_bipedal.float()  # First column for quadrupedal state
        obs_buf[:, 1] = self.is_three_foot.float()  # Second column for three-legged state
        obs_buf[:, 2] = (~self.is_bipedal & ~self.is_three_foot).float()  # Third column for bipedal state

        return obs_buf

    def _write_one_hot_noise(self, noise_vec):
        """ No noise for one hot encoding. """
        noise_vec[:] = 0.  # no noise for one hot encoding
    
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "proprioception" in components:
            segments["proprioception"] = (48,)
        if "one_hot" in components:
            segments["one_hot"] = (3,)
        if "proprio_with_goal" in components:
            segments["proprio_with_goal"] = (49,)
        if "height_measurements" in components:
            segments["height_measurements"] = (len(self.cfg.terrain.measured_points_x)*len(self.cfg.terrain.measured_points_y),)
        if "forward_depth" in components:
            segments["forward_depth"] = (1, *self.cfg.sensor.forward_camera.resolution)
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            segments["robot_config"] = (1 + 3 + 1 + 12,)
        if "robot_friction" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            segments["robot_friction"] = (1,)
        if "stumble_state" in components:
            num_dim = 0
            if "foot" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "hip" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "thigh" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "calf" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "base" in self.cfg.asset.collision_body_names:
                num_dim += 1
            segments["stumble_state"] = (num_dim,)
        return segments