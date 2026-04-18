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

class LeggedRobotMix(LeggedRobotField):
    """ Using inheritance to ensure clean mixture of two environments: quadrupedal and bipedal.
    """
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        print("Using LeggedRobotMix.__init__, num_obs and num_privileged_obs will be computed instead of assigned.")
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        is_bipedal_default = self.root_states[:, 1] <= self.cfg.terrain.terrain_width * (self.cfg.terrain.bipedal_cols)
    
        if getattr(self.cfg.commands, "switch_time", 0) > 0:
            if max(self.cfg.commands.switch_cols) >= self.cfg.terrain.bipedal_cols:
                raise ValueError("switch_cols exceed bipedal_cols-1")
            
            col_indices = torch.floor(self.root_states[:, 1] / self.cfg.terrain.terrain_width).long()
            for col in self.cfg.commands.switch_cols:
                is_bipedal_default = torch.where(col_indices == col,
                                                torch.zeros_like(is_bipedal_default, dtype=torch.bool),
                                                is_bipedal_default)
        self.is_bipedal = is_bipedal_default
        
    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()

        if getattr(self.cfg.commands, "switch_time", 0) > 0:
            col_indices = torch.floor(self.root_states[:, 1] / self.cfg.terrain.terrain_width).long()
            is_bipedal_new = self.root_states[:, 1] <= self.cfg.terrain.terrain_width * self.cfg.terrain.bipedal_cols
            switch_time_steps = int(self.cfg.commands.switch_time / self.dt)
            for col in self.cfg.commands.switch_cols:
                mask = col_indices == col
                is_bipedal_new = torch.where(mask,
                                            self.episode_length_buf >= switch_time_steps,
                                            is_bipedal_new)
            self.is_bipedal = is_bipedal_new
        else:
            self.is_bipedal = self.root_states[:, 1] <= self.cfg.terrain.terrain_width * self.cfg.terrain.bipedal_cols

        self._resample_commands(env_ids)
        
        # Process heading commands if enabled.
        if self.cfg.commands.heading_command:
            # Compute the forward direction in the world frame using the base quaternion.
            forward = quat_apply(self.base_quat, self.forward_vec)
            # Calculate the current heading angle.
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            
            if self.cfg.terrain.unify:
                # Compute column indices based on y-position.
                col_indices = torch.floor(self.root_states[:, 1] / self.cfg.terrain.terrain_width).long()
                # Build a mask for environments in the unified columns.
                mask = torch.zeros_like(col_indices, dtype=torch.bool)
                for col in self.cfg.terrain.unify_cols:
                    mask |= (col_indices == col)
                        
                new_heading_cmd = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
                self.commands[:, 2] = torch.where(mask, new_heading_cmd, self.commands[:, 2])
            else:
                self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
            
            if self.cfg.terrain.unify:
                norm_mask = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
                self.commands[:, 2] = torch.where(mask, self.commands[:, 2] * norm_mask, self.commands[:, 2])
                
        # --- Zero the commands before switching ---
        if getattr(self.cfg.commands, "switch_time", 0) > 0:
            col_indices = torch.floor(self.root_states[:, 1] / self.cfg.terrain.terrain_width).long()
            switch_mask = torch.zeros_like(col_indices, dtype=torch.bool)
            for col in self.cfg.commands.switch_cols:
                # The condition:
                #   1. The environment's column equals the current column.
                #   2. The robot is still quadrupedal (is_bipedal is False).
                #   3. The elapsed time (episode_length_buf * dt) is greater than (switch_time - 1.0) seconds.
                condition = (col_indices == col) & (~self.is_bipedal) & ((self.episode_length_buf * self.dt) > (self.cfg.commands.switch_time - 1.0))
                switch_mask |= condition
            self.commands[switch_mask, :] = 0.0
        
        # Log maximum power over the current timestep.
        self.max_power_per_timestep = torch.maximum(
            self.max_power_per_timestep,
            torch.max(torch.sum(self.substep_torques * self.substep_dof_vel, dim=-1), dim=-1)[0],
        )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
            
        
    def _resample_commands(self, env_ids):
        """
        Randomly selects new commands for a subset of environments. Some environments may be bipedal,
        in which case they sample different command ranges than non-bipedal environments.

        Args:
            env_ids (List[int]): A list of environment IDs for which new commands need to be generated.
        """
        # Boolean mask indicating which environments are bipedal; shape = (len(env_ids),).
        is_bipedal = self.is_bipedal[env_ids]

        # 1. Resample commands for linear velocity in x-direction (index 0).
        self.commands[env_ids, 0] = torch_rand_tensor(
            torch.where(
                is_bipedal,
                torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_x"][0], device=self.device),
                torch.full((len(env_ids),), self.command_ranges["lin_vel_x"][0], device=self.device),
            ),
            torch.where(
                is_bipedal,
                torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_x"][1], device=self.device),
                torch.full((len(env_ids),), self.command_ranges["lin_vel_x"][1], device=self.device),
            ),
            (len(env_ids),),
            device=self.device
        )

        # 2. Resample commands for linear velocity in y-direction (index 1).
        self.commands[env_ids, 1] = torch_rand_tensor(
            torch.where(
                is_bipedal,
                torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_y"][0], device=self.device),
                torch.full((len(env_ids),), self.command_ranges["lin_vel_y"][0], device=self.device),
            ),
            torch.where(
                is_bipedal,
                torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_y"][1], device=self.device),
                torch.full((len(env_ids),), self.command_ranges["lin_vel_y"][1], device=self.device),
            ),
            (len(env_ids),),
            device=self.device
        )

        # 3. Resample commands for heading (index 3).
        self.commands[env_ids, 3] = torch_rand_tensor(
            torch.full((len(env_ids),), self.command_ranges["heading"][0], device=self.device),
            torch.full((len(env_ids),), self.command_ranges["heading"][1], device=self.device),
            (len(env_ids),),
            device=self.device
        )

        # 4. Resample commands for angular velocity in yaw (index 2).
        self.commands[env_ids, 2] = torch_rand_tensor(
            torch.where(
                is_bipedal,
                torch.full((len(env_ids),), self.command_ranges["bipedal_ang_vel_yaw"][0], device=self.device),
                torch.full((len(env_ids),), self.command_ranges["ang_vel_yaw"][0], device=self.device),
            ),
            torch.where(
                is_bipedal,
                torch.full((len(env_ids),), self.command_ranges["bipedal_ang_vel_yaw"][1], device=self.device),
                torch.full((len(env_ids),), self.command_ranges["ang_vel_yaw"][1], device=self.device),
            ),
            (len(env_ids),),
            device=self.device
        )
        
        # If unified training is enabled, override commands based on terrain column.
        if self.cfg.terrain.unify:
            # Compute the column indices based on the y-position of each environment.
            # Each column's width is defined by self.cfg.terrain.terrain_width.
            col_indices = torch.floor(self.root_states[env_ids, 1] / self.cfg.terrain.terrain_width).long()
            
            # Create a boolean mask where True indicates that the environment's column is in unify_cols.
            mask = torch.zeros_like(col_indices, dtype=torch.bool)
            for col in self.cfg.terrain.unify_cols:
                mask |= (col_indices == col)

            # For environments in unified columns, we sample new commands based on whether they are bipedal.
            # For lin_vel_x: if bipedal, use the bipedal command range; otherwise, use the standard command range.
            new_lin_vel_x = torch_rand_tensor(
                torch.where(
                    is_bipedal,
                    torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_x"][0], device=self.device),
                    torch.full((len(env_ids),), self.command_ranges["lin_vel_x"][0], device=self.device)
                ),
                torch.where(
                    is_bipedal,
                    torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_x"][1], device=self.device),
                    torch.full((len(env_ids),), self.command_ranges["lin_vel_x"][1], device=self.device)
                ),
                (len(env_ids),),
                device=self.device
            )

            # For lin_vel_y: if bipedal, sample from the bipedal command range; otherwise, set the command to 0.
            new_lin_vel_y = torch_rand_tensor(
                torch.where(
                    is_bipedal,
                    torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_y"][0], device=self.device),
                    torch.full((len(env_ids),), 0.0, device=self.device)
                ),
                torch.where(
                    is_bipedal,
                    torch.full((len(env_ids),), self.command_ranges["bipedal_lin_vel_y"][1], device=self.device),
                    torch.full((len(env_ids),), 0.0, device=self.device)
                ),
                (len(env_ids),),
                device=self.device
            )

            # For ang_vel_yaw: if bipedal, sample from the bipedal command range; otherwise, set the command to 0.
            new_ang_vel_yaw = torch_rand_tensor(
                torch.where(
                    is_bipedal,
                    torch.full((len(env_ids),), self.command_ranges["bipedal_ang_vel_yaw"][0], device=self.device),
                    torch.full((len(env_ids),), 0.0, device=self.device)
                ),
                torch.where(
                    is_bipedal,
                    torch.full((len(env_ids),), self.command_ranges["bipedal_ang_vel_yaw"][1], device=self.device),
                    torch.full((len(env_ids),), 0.0, device=self.device)
                ),
                (len(env_ids),),
                device=self.device
            )

            # Update the commands for environments in unified columns:
            # Override lin_vel_x, lin_vel_y, and ang_vel_yaw with the newly sampled commands.
            self.commands[env_ids, 0] = torch.where(mask, new_lin_vel_x, self.commands[env_ids, 0])
            self.commands[env_ids, 1] = torch.where(mask, new_lin_vel_y, self.commands[env_ids, 1])
            self.commands[env_ids, 2] = torch.where(mask, new_ang_vel_yaw, self.commands[env_ids, 2])

        # 5. Zero out small linear velocity commands based on a threshold.
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) >
            torch.where(
                is_bipedal,
                torch.tensor(self.cfg.commands.bipedal_clear_vel_cmd_threshold, device=self.device),
                torch.tensor(self.cfg.commands.clear_vel_cmd_threshold, device=self.device)
            )
        ).unsqueeze(1)

        # 6. Zero out small yaw commands based on a threshold.
        self.commands[env_ids, 2] *= (
            torch.abs(self.commands[env_ids, 2]) >
            torch.where(
                is_bipedal,
                torch.tensor(self.cfg.commands.bipedal_clear_ang_cmd_threshold, device=self.device),
                torch.tensor(self.cfg.commands.clear_ang_cmd_threshold, device=self.device)
            )
        )
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if hasattr(self.cfg.domain_rand, "init_base_pos_range"):
                self.root_states[env_ids, 0:1] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["x"], (len(env_ids), 1), device=self.device)
                self.root_states[env_ids, 1:2] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["y"], (len(env_ids), 1), device=self.device)
                self.root_states[env_ids, 2:3] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["z"], (len(env_ids), 1), device=self.device)
            else:
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation (roll, pitch, yaw)
        if hasattr(self.cfg.domain_rand, "init_base_rot_range"):
            base_roll = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_pitch = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            if self.cfg.terrain.unify:
                col_indices = torch.floor(self.root_states[env_ids, 1] / self.cfg.terrain.terrain_width).long()

                mask = torch.zeros_like(col_indices, dtype=torch.bool)
                for col in self.cfg.terrain.unify_cols:
                    mask |= (col_indices == col)

                base_yaw_unified = torch_rand_float(
                    *([-0.2, 0.2]),
                    (len(env_ids), 1),
                    device=self.device,
                )[:, 0]

                base_yaw_default = torch_rand_float(
                    *self.cfg.domain_rand.init_base_rot_range["yaw"],
                    (len(env_ids), 1),
                    device=self.device,
                )[:, 0]

                base_yaw = base_yaw_unified * mask.float() + base_yaw_default * (~mask).float()
            else:
                base_yaw = torch_rand_float(
                    *self.cfg.domain_rand.init_base_rot_range["yaw"],
                    (len(env_ids), 1),
                    device=self.device,
                )[:, 0]
            base_quat = quat_from_euler_xyz(base_roll, base_pitch, base_yaw)
            self.root_states[env_ids, 3:7] = base_quat
        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        if isinstance(base_vel_range, (tuple, list)):
            if self.cfg.terrain.unify:
                col_indices = torch.floor(self.root_states[env_ids, 1] / self.cfg.terrain.terrain_width).long()
                mask = torch.zeros_like(col_indices, dtype=torch.bool)
                for col in self.cfg.terrain.unify_cols:
                    mask |= (col_indices == col)
                
                new_base_vel = torch_rand_float(
                    *base_vel_range,
                    (len(env_ids), 6),
                    device=self.device,
                )
                zeros = torch.zeros((len(env_ids), 6), device=self.device)
                
                self.root_states[env_ids, 7:13] = torch.where(mask.unsqueeze(1), new_base_vel, zeros)
            else:
                self.root_states[env_ids, 7:13] = torch_rand_float(
                    *base_vel_range,
                    (len(env_ids), 6),
                    device=self.device,
                )
        elif isinstance(base_vel_range, dict):
            self.root_states[env_ids, 7:8] = torch_rand_float(
                *base_vel_range["x"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 8:9] = torch_rand_float(
                *base_vel_range["y"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 9:10] = torch_rand_float(
                *base_vel_range["z"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 10:11] = torch_rand_float(
                *base_vel_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 11:12] = torch_rand_float(
                *base_vel_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 12:13] = torch_rand_float(
                *base_vel_range["yaw"],
                (len(env_ids), 1),
                device=self.device,
            )
        else:
            raise NameError(f"Unknown base_vel_range type: {type(base_vel_range)}")
        
        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        actor_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        actor_idx_int32 = actor_idx.to(dtype=torch.int32)
        # self.root_states[0, 2]=0.23
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))
    
    def _push_robots(self):
        """Randomly pushes the robots by applying an impulse with a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy

        if self.cfg.terrain.unify:
            col_indices = torch.floor(self.root_states[:, 1] / self.cfg.terrain.terrain_width).long()
            mask = torch.zeros_like(col_indices, dtype=torch.bool)
            for col in self.cfg.terrain.unify_cols:
                mask |= (col_indices == col)
            
            mask_expanded = mask.unsqueeze(1).float()  # Shape: (num_envs, 1)

            random_push = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
            
            self.root_states[:, 7:9] = random_push * mask_expanded + self.root_states[:, 7:9] * (1 - mask_expanded)
        else:
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))
    
    def update_track_rew(self):
        #compute quadrupedal tracking
        quad_lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        quad_rew = torch.exp(-quad_lin_vel_error/self.cfg.rewards.tracking_sigma)
        
        #compute bipedal tracking
        actual_lin_vel = quat_apply_yaw_inverse(self.base_quat, self.root_states[:, 7:10])
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - actual_lin_vel[:, :2]), dim=1)
        bipedal_reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

        # Compute Weight
        target_vector = torch.tensor([-1.0, 0.0, 0.0], device=self.projected_gravity.device)
        target_vector = target_vector.unsqueeze(0).repeat(self.projected_gravity.shape[0], 1)  # (N, 3)

        dot_product = torch.sum(self.projected_gravity * target_vector, dim=1)  # (N,)
        gravity_norm = torch.norm(self.projected_gravity, dim=1)  # (N,)
        target_norm = torch.norm(target_vector, dim=1)  # (N,)
        cos_theta = dot_product / (gravity_norm * target_norm + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  
        theta = torch.arccos(cos_theta) 

        # Reward calculation with the condition
        # weight = torch.where(
        #     theta < self.cfg.rewards.bipedal_theta_threshold,
        #     torch.ones_like(theta),  # Reward is 1 when theta < 0.3
        #     torch.exp(-theta / self.cfg.rewards.tracking_sigma)  # Exponential decay otherwise
        # )
        is_stand = cos_theta > 0.95
        # Compute reward using exponential function
        scale_factor_low = self.cfg.rewards.scale_factor_low
        scale_factor_high = self.cfg.rewards.scale_factor_high
        scaling_factor = (torch.clip(
            self.root_states[:, 2], min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        
        bipedal_reward = bipedal_reward * is_stand.float() * scaling_factor
        
        self.track_lin_vel_buf += quad_rew * ~self.is_bipedal + bipedal_reward * self.is_bipedal
        self.terrain_time += 1
        
    def _get_terrain_curriculum_move(self, env_ids):
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)

        is_bipedal = self.is_bipedal[env_ids]

        move_up_bipedal = (self.track_lin_vel_buf[env_ids] / self.terrain_time[env_ids] > 0.5) * (distance > self.terrain.env_length / 2 - 3.5)
        move_down_bipedal = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.4) * \
                            (distance < self.terrain.env_length / 2 - 3.5) * (~move_up_bipedal)

        move_up_non_bipedal = (self.track_lin_vel_buf[env_ids] / self.terrain_time[env_ids] > 0.75) * (distance > self.terrain.env_length / 2 - 1)
        move_down_non_bipedal = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * \
                                (distance < self.terrain.env_length / 2 - 1) * (~move_up_non_bipedal)

        move_up = torch.where(is_bipedal, move_up_bipedal, move_up_non_bipedal)
        move_down = torch.where(is_bipedal, move_down_bipedal, move_down_non_bipedal)

        return move_up, move_down
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # self.reset_buf |= self.time_out_buf
        self.reset_buf=self.time_out_buf.clone()
        
        r, p, y = get_euler_xyz(self.base_quat)
        r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
        p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
        z = self.root_states[:, 2] - self.env_origins[:, 2]

        if getattr(self.cfg.termination, "check_obstacle_conditioned_threshold", False) and self.check_BarrierTrack_terrain():
            if hasattr(self, "volume_sample_points"):
                self.refresh_volume_sample_points()
                stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.volume_sample_points.view(-1, 3))
            else:
                stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.root_states[:, :3])
            stepping_obstacle_info = stepping_obstacle_info.view(self.num_envs, -1, stepping_obstacle_info.shape[-1])
            # Assuming that each robot will only be in one obstacle or non obstacle.
            robot_stepping_obstacle_id = torch.max(stepping_obstacle_info[:, :, 0], dim= -1)[0]
        
        if "roll" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                r_term_buff = torch.abs(r[robot_stepping_obstacle_id == 0]) > \
                    self.cfg.termination.roll_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= r_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.roll_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        r_term_buff = torch.abs(r[env_selection_mask]) > \
                            self.cfg.termination.roll_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= r_term_buff
            else:
                r_term_buff = torch.abs(r) > self.cfg.termination.roll_kwargs["threshold"]
                self.reset_buf |= r_term_buff * ~self.is_bipedal
                # print("roll")
                
        if "pitch" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                p_term_buff = torch.abs(p[robot_stepping_obstacle_id == 0]) > \
                    self.cfg.termination.pitch_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= p_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.pitch_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        p_term_buff = torch.abs(p[env_selection_mask]) > \
                            self.cfg.termination.pitch_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= p_term_buff
            else:
                p_term_buff = torch.abs(p) > self.cfg.termination.pitch_kwargs["threshold"]
                self.reset_buf |= p_term_buff * ~self.is_bipedal
                # print("pitch")
            
        if "contact_force" in self.cfg.termination.termination_terms:
            allowed_steps = self.cfg.rewards.allow_contact_steps
            if getattr(self.cfg.commands, "switch_time", 0) > 0:
                allowed_steps += int(self.cfg.commands.switch_time / self.dt)
            # Check if the episode length exceeds allow_contact_steps
            episode_length_criteria = self.episode_length_buf > allowed_steps
            
            # Calculate body contact based on the monitored collision bodies
            body_contact = (torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0)  # Shape: (N, num_collision_bodies)

            # Determine if any monitored body exceeds the contact force threshold
            force_exceed_criteria = torch.any(body_contact, dim=1)  # Shape: (N,)
            termination_criteria = episode_length_criteria & force_exceed_criteria
            self.reset_buf |= termination_criteria * self.is_bipedal
            
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function that had a non-zero scale (processed in self._prepare_reward_function())
            and adds each term to the episode sums and to the total reward.
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # If the reward name starts with "bipedal", include it only if self.is_bipedal is True;
            # otherwise, include it only if self.is_bipedal is False.
            if name.startswith("bipedal"):
                self.rew_buf += rew * self.is_bipedal
                self.episode_sums[name] += rew * self.is_bipedal
            else:
                self.rew_buf += rew * ~self.is_bipedal
                self.episode_sums[name] += rew * ~self.is_bipedal
            

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        # Process termination reward after clipping.
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            
    def _reward_bipedal_rear_air(self):
        contact = self.contact_forces[:, self.rear_feet_indices, 2] < 1.
        # init_condition = self.root_states[:, 2] < 0.3
        # init_condition = self.episode_length_buf < self.cfg.rewards.allow_contact_steps
        # reward = (torch.all(contact, dim=1) * (~init_condition) + torch.any(contact, dim=1) * init_condition).float()
        calf_contact = self.contact_forces[:, self.rear_calf_indices, 2] < 1.
        unhealthy_condition = torch.logical_and(~calf_contact, contact)
        reward = torch.all(contact, dim=1).float() + unhealthy_condition.sum(dim=-1).float()
        return reward


    def _reward_bipedal_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_bipedal_front_hip_pos(self):
        """ Reward the robot to stop moving its front hips """
        return torch.sum(torch.square(self.dof_pos[:, self.front_hip_indices] - self.default_dof_pos[:, self.front_hip_indices]), dim=1)

    def _reward_bipedal_rear_hip_pos(self):
        """ Reward the robot to stop moving its rear hips """
        return torch.sum(torch.square(self.dof_pos[:, self.rear_hip_indices] - self.default_dof_pos[:, self.rear_hip_indices]), dim=1)
    
    def _reward_bipedal_rear_pos_balance(self):
        """ Reward the robot to stop moving its rear hips """
        reward = torch.sum(torch.square(self.dof_pos[:, self.rear_left_dof_indices] - self.dof_pos[:, self.rear_right_dof_indices]), dim=1)
        return reward
    
    def _reward_bipedal_front_dof_pos(self):
        """ Reward the robot to stop moving its front dofs """
        reward = torch.sum(torch.square(self.dof_pos[:, self.front_dof_indices] - self.default_dof_pos[:, self.front_dof_indices]), dim=1)
        reward *= (self.episode_length_buf > self.cfg.rewards.allow_contact_steps).float()
        return reward
    
    def _reward_bipedal_front_dof_vel(self):
        # Penalize front dof velocities
        reward = torch.sum(torch.square(self.dof_vel[:, self.front_dof_indices]), dim=1)
        reward *= (self.episode_length_buf > self.cfg.rewards.allow_contact_steps).float()
        return reward
    
    def _reward_bipedal_front_dof_acc(self):
        # Penalize front dof accelerations
        reward = torch.sum(torch.square((self.last_dof_vel[:, self.front_dof_indices] - self.dof_vel[:, self.front_dof_indices]) / self.dt), dim=1)
        reward *= (self.episode_length_buf > self.cfg.rewards.allow_contact_steps).float()
        return reward
    
    def _reward_bipedal_legs_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(self.substep_torques * self.substep_dof_vel), dim=-1), dim=-1)
    
    def _reward_bipedal_alive(self):
        return 1.
    
    def _reward_bipedal_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - self.torque_limits
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p= 1, dim= -1).sum(dim= 1)
    
    def _reward_bipedal_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim= -1).mean(dim= -1)
    
    def _reward_bipedal_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0), dim=1)
    
    def _reward_bipedal_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_bipedal_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_bipedal_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
 
    def _reward_bipedal_tracking_lin_vel(self):
        actual_lin_vel = quat_apply_yaw_inverse(self.base_quat, self.root_states[:, 7:10])
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - actual_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.rewards.bipedal_tracking_sigma)

        # Compute Weight
        target_vector = torch.tensor([-1.0, 0.0, 0.0], device=self.projected_gravity.device)
        target_vector = target_vector.unsqueeze(0).repeat(self.projected_gravity.shape[0], 1)  # (N, 3)

        dot_product = torch.sum(self.projected_gravity * target_vector, dim=1)  # (N,)
        gravity_norm = torch.norm(self.projected_gravity, dim=1)  # (N,)
        target_norm = torch.norm(target_vector, dim=1)  # (N,)
        cos_theta = dot_product / (gravity_norm * target_norm + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  
        theta = torch.arccos(cos_theta) 

        # Reward calculation with the condition
        # weight = torch.where(
        #     theta < self.cfg.rewards.bipedal_theta_threshold,
        #     torch.ones_like(theta),  # Reward is 1 when theta < 0.3
        #     torch.exp(-theta / self.cfg.rewards.tracking_sigma)  # Exponential decay otherwise
        # )
        is_stand = cos_theta > 0.95
        # Compute reward using exponential function
        scale_factor_low = self.cfg.rewards.scale_factor_low
        scale_factor_high = self.cfg.rewards.scale_factor_high
        scaling_factor = (torch.clip(
            self.root_states[:, 2], min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        
        reward = reward * is_stand.float() * scaling_factor
        
        return reward
    
    def _reward_bipedal_tracking_ang_vel(self):
        # use roll to compute ang vel err
        bipedal_ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 1])
        reward = torch.exp(-bipedal_ang_vel_error/self.cfg.rewards.bipedal_tracking_ang_sigma)
        
        # use heading to compute ang vel err
        # heading = self._get_cur_heading()
        # heading_error = torch.square(wrap_to_pi(self.commands[:, 3] - heading) / np.pi)
        # reward = torch.exp(-heading_error / self.cfg.rewards.tracking_ang_sigma)

        # Compute Weight
        target_vector = torch.tensor([-1.0, 0.0, 0.0], device=self.projected_gravity.device)
        target_vector = target_vector.unsqueeze(0).repeat(self.projected_gravity.shape[0], 1)  # (N, 3)

        dot_product = torch.sum(self.projected_gravity * target_vector, dim=1)  # (N,)
        gravity_norm = torch.norm(self.projected_gravity, dim=1)  # (N,)
        target_norm = torch.norm(target_vector, dim=1)  # (N,)
        cos_theta = dot_product / (gravity_norm * target_norm + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  
        theta = torch.arccos(cos_theta) 

        # # Reward calculation with the condition
        # weight = torch.where(
        #     theta < 0.3,
        #     torch.ones_like(theta),  # Reward is 1 when theta < 0.3
        #     torch.exp(-theta / self.cfg.rewards.tracking_sigma)  # Exponential decay otherwise
        # )
        is_stand = cos_theta > 0.95
        # Compute reward using exponential function
        scale_factor_low = self.cfg.rewards.scale_factor_low
        scale_factor_high = self.cfg.rewards.scale_factor_high
        scaling_factor = (torch.clip(
            self.root_states[:, 2], min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        
        reward = reward * is_stand.float() * scaling_factor
        
        
        return reward
    
    
    
    def _get_one_hot_obs(self, privileged= False):
        # with dim 1
        # 0 means quadrupedal, 1 means bipedal
        obs_buf = self.is_bipedal.float().unsqueeze(-1)
        return obs_buf
    
    def _write_one_hot_noise(self, noise_vec):
        noise_vec[:] = 0. # no noise for one hot encoding
        
        
def torch_rand_tensor(lower: torch.Tensor, 
                      upper: torch.Tensor,
                      shape: tuple,
                      device: str = "cpu") -> torch.Tensor:
    """
    Generates random numbers in the range [lower, upper], supporting lower and upper as Tensors.
    The returned Tensor will have the specified shape, which must be broadcast-compatible with
    the shapes of 'lower' and 'upper' according to PyTorch broadcasting rules.

    Args:
        lower (torch.Tensor): Lower bound(s). Must broadcast to 'shape'.
        upper (torch.Tensor): Upper bound(s). Must broadcast to 'shape'.
        shape (tuple): Shape of the output random Tensor.
        device (str): The device on which to place the resulting Tensor.

    Returns:
        torch.Tensor: A Tensor of random values in the range [lower, upper], with shape 'shape'.
    """
    # Generate a uniform random tensor on [0, 1], then scale and shift to [lower, upper].
    return lower + (upper - lower) * torch.rand(*shape, device=device)