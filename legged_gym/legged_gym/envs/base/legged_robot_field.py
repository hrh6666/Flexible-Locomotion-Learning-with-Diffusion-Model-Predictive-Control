from collections import OrderedDict, defaultdict
import itertools
import numpy as np
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import get_terrain_cls
from legged_gym.utils.observation import get_obs_slice
from .legged_robot_config import LeggedRobotCfg

class LeggedRobotField(LeggedRobot):
    """ NOTE: Most of this class implementation does not depend on the terrain. Check where
    `check_BarrierTrack_terrain` is called to remove the dependency of BarrierTrack terrain.
    """
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        print("Using LeggedRobotField.__init__, num_obs and num_privileged_obs will be computed instead of assigned.")
        cfg.terrain.measure_heights = True # force height measurement that have full obs from parent class implementation.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # --- RND ---------------------------------------------------------------------
        in_dim  = self.num_dof * 2
        embed_dim = self.cfg.rewards.rnd_embed_dim if hasattr(self.cfg.rewards, "rnd_embed_dim") else 128
        lr_rnd = self.cfg.rewards.lr_rnd if hasattr(self.cfg.rewards, "lr_rnd") else 1e-4

        class _RNDModel(nn.Module):
            """Simple MLP used for both target and predictor."""
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 128), nn.ReLU(),
                    nn.Linear(128, 128),  nn.ReLU(),
                    nn.Linear(128, out_dim)
                )

            def forward(self, x):
                return self.net(x)

        # target network (f) – fixed during training
        self.rnd_target = _RNDModel(in_dim, embed_dim).to(self.device)
        for p in self.rnd_target.parameters():
            p.requires_grad_(False)

        # predictor network (f̂) – trainable
        self.rnd_predictor = _RNDModel(in_dim, embed_dim).to(self.device)
        self.rnd_opt = torch.optim.Adam(self.rnd_predictor.parameters(), lr=lr_rnd)
        # ----------------------------------------------------------------------------- 
        
    def check_BarrierTrack_terrain(self):
        if getattr(self.cfg.terrain, "pad_unavailable_info", False):
            return self.cfg.terrain.selected == "BarrierTrack"
        assert self.cfg.terrain.selected == "BarrierTrack", "This implementation is only for BarrierTrack terrain"
        return True

    ##### Working on simulation steps #####
    def pre_physics_step(self, actions):
        self.volume_sample_points_refreshed = False        
        return super().pre_physics_step(actions)
    
    def check_termination(self):
        return_ = super().check_termination()
        if not hasattr(self.cfg, "termination"): return return_
        
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
                self.reset_buf |= r_term_buff
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
                self.reset_buf |= p_term_buff
                # print("pitch")
        if "z_low" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                z_low_term_buff = z[robot_stepping_obstacle_id == 0] < \
                    self.cfg.termination.z_low_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= z_low_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.z_low_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        z_low_term_buff = z[env_selection_mask] < \
                            self.cfg.termination.z_low_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= z_low_term_buff
            else:
                z_low_term_buff = z < self.cfg.termination.z_low_kwargs["threshold"]
                self.reset_buf |= z_low_term_buff
        if "z_high" in self.cfg.termination.termination_terms:
            z_high_term_buff = z > self.cfg.termination.z_high_kwargs["threshold"]
            self.reset_buf |= z_high_term_buff
        if "out_of_track" in self.cfg.termination.termination_terms and self.check_BarrierTrack_terrain():
            # robot considered dead if it goes side ways
            side_distance = self.terrain.get_sidewall_distance(self.root_states[:, :3])
            side_diff = torch.abs(side_distance[..., 0] - side_distance[..., 1])
            out_of_track_buff = side_diff > self.cfg.termination.out_of_track_kwargs["threshold"]
            out_of_track_buff |= side_distance[..., 0] <= 0
            out_of_track_buff |= side_distance[..., 1] <= 0
            self.reset_buf |= out_of_track_buff
        if "stuck" in self.cfg.termination.termination_terms:
            # robot considered dead if it stuck even the command is large.
            now_pos=self.root_states[:, :2]
            now_vel = torch.norm(self.base_lin_vel[:, :2],dim=1)
            #goal_dis = torch.norm(self.delta_goal[:, :2], dim=1)
            # distance = torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1)
            # now_command = torch.norm(self.commands[:, :2], dim=1)
            self.recover_buf += (now_vel > self.cfg.termination.stuck_kwargs["threshold"])
            self.stuck_buf += (now_vel < self.cfg.termination.stuck_kwargs["threshold"])
            self.recover_buf *= (now_vel > self.cfg.termination.stuck_kwargs["threshold"])
            self.stuck_buf *= self.recover_buf<self.cfg.termination.stuck_kwargs["recover_steps"]
            stuck_buff = self.stuck_buf > self.cfg.termination.stuck_kwargs["steps"]
            self.reset_buf |= stuck_buff
            
            # stuck_buff=(now_command > 0.5)*(self.episode_length_buf>150)*(distance<1)
            # stuck_buff= (self.track_lin_vel_buf/self.terrain_time<0.65)*(self.episode_length_buf>100)
            
            # self.reset_buf |= stuck_buff
            
        if "contact_force" in self.cfg.termination.termination_terms:
            # Check if the episode length exceeds allow_contact_steps
            episode_length_criteria = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
            
            # Calculate body contact based on the monitored collision bodies
            body_contact = (torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0)  # Shape: (N, num_collision_bodies)

            # Determine if any monitored body exceeds the contact force threshold
            force_exceed_criteria = torch.any(body_contact, dim=1)  # Shape: (N,)
            termination_criteria = episode_length_criteria & force_exceed_criteria
            self.reset_buf |= termination_criteria

        if getattr(self.cfg.termination, "timeout_at_border", False):
            track_idx = self.terrain.get_track_idx(self.root_states[:, :3], clipped= False)
            # The robot is going +x direction, so no checking for row_idx <= 0
            out_of_border_buff = track_idx[:, 0] >= self.terrain.cfg.num_rows
            out_of_border_buff |= track_idx[:, 1] < 0
            out_of_border_buff |= track_idx[:, 1] >= self.terrain.cfg.num_cols

            self.time_out_buf |= out_of_border_buff
            self.reset_buf |= out_of_border_buff
        if getattr(self.cfg.termination, "timeout_at_finished", False):
            x = self.root_states[:, 0] - self.env_origins[:, 0]
            y = self.root_states[:, 1] - self.env_origins[:, 1]
            # finished_buffer = x > (self.terrain.env_block_length * self.terrain.n_blocks_per_track)
            finished_buffer_x = torch.abs(x) > (self.cfg.terrain.terrain_width/2-0.7)
            finished_buffer_y = torch.abs(y) > (self.cfg.terrain.terrain_length/2-0.7)
            finished_buffer=finished_buffer_x|finished_buffer_y
            self.time_out_buf |= finished_buffer
            self.reset_buf |= finished_buffer
        
        return return_

    def _fill_extras(self, env_ids):
        return_ = super()._fill_extras(env_ids)

        self.extras["episode"]["n_obstacle_passed"] = 0.
        with torch.no_grad():
            pos_x = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
            self.extras["episode"]["pos_x"] = pos_x
            if self.check_BarrierTrack_terrain():
                self.extras["episode"]["n_obstacle_passed"] = torch.mean(torch.clip(
                    torch.div(pos_x, self.terrain.env_block_length, rounding_mode= "floor") - 1,
                    min= 0.0,
                )).cpu()
        
        return return_

    def _post_physics_step_callback(self):
        return_ = super()._post_physics_step_callback()

        with torch.no_grad():
            pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
            pos_y = self.root_states[:, 1] - self.env_origins[:, 1]
            if self.check_BarrierTrack_terrain():
                self.extras["episode"]["n_obstacle_passed"] = torch.mean(torch.clip(
                    torch.div(pos_x, self.terrain.env_block_length, rounding_mode= "floor") - 1,
                    min= 0.0,
                )).cpu()

        return return_
    
    def _get_terrain_curriculum_move(self, env_ids):
        if not (self.cfg.terrain.selected == "BarrierTrack" and self.cfg.terrain.BarrierTrack_kwargs["virtual_terrain"] and hasattr(self, "body_sample_indices")):
            if getattr(self.cfg.curriculum, "no_moveup_when_fall", False):
                move_up, move_down = super()._get_terrain_curriculum_move(env_ids)
                move_up = move_up & self.time_out_buf[env_ids]
                return move_up, move_down
            else:
                return super()._get_terrain_curriculum_move(env_ids)
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        moved = distance > (self.terrain.env_block_length * 1.5) # 0.1 is the guess of robot touching the obstacle block.
        passed_depths = self.terrain.get_passed_obstacle_depths(
            self.terrain_levels[env_ids],
            self.terrain_types[env_ids],
            self.volume_sample_points[env_ids, :, 0].max(-1)[0], # choose the sample points that goes the furthest
        ) + 1e-12

        p_v_ok = p_d_ok = 1
        p_v_too_much = p_d_too_much = 0
        # NOTE: only when penetrate_* reward is computed does this function check the penetration
        if "penetrate_volume" in self.episode_sums:
            p_v = self.episode_sums["penetrate_volume"][env_ids]
            p_v_normalized = p_v / passed_depths / self.reward_scales["penetrate_volume"]
            p_v_ok = p_v_normalized < self.cfg.curriculum.penetrate_volume_threshold_harder
            p_v_too_much = p_v_normalized > self.cfg.curriculum.penetrate_volume_threshold_easier
        if "penetrate_depth" in self.episode_sums:
            p_d = self.episode_sums["penetrate_depth"][env_ids]
            p_d_normalized = p_d / passed_depths / self.reward_scales["penetrate_depth"]
            p_d_ok = p_d_normalized < self.cfg.curriculum.penetrate_depth_threshold_harder
            p_d_too_much = p_d_normalized > self.cfg.curriculum.penetrate_depth_threshold_easier

        # print("p_v:", p_v_normalized, "p_d:", p_d_normalized)
        move_up = p_v_ok * p_d_ok * moved
        move_down = ((~moved) + p_v_too_much + p_d_too_much).to(bool)
        return move_up, move_down

    ##### Dealing with observations #####
    def _init_buffers(self):
        # update obs_scales components incase there will be one-by-one scaling
        for k in self.all_obs_components:
            if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
                setattr(
                    self.obs_scales,
                    k,
                    torch.tensor(getattr(self.obs_scales, k, 1.), dtype= torch.float32, device= self.device)
                )
        
        super()._init_buffers()

        # projected gravity bias (if needed)
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            print("Initializing gravity bias for domain randomization")
            # add cross trajectory domain randomization on projected gravity bias
            # uniform sample from range
            self.gravity_bias = torch.rand(self.num_envs, 3, dtype= torch.float, device= self.device, requires_grad= False)
            self.gravity_bias[:, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[:, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]

    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)
        if hasattr(self, "velocity_sample_points"): self.velocity_sample_points[env_ids] = 0.

        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            assert hasattr(self, "gravity_bias")
            self.gravity_bias[env_ids] = torch.rand_like(self.gravity_bias[env_ids])
            self.gravity_bias[env_ids, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[env_ids, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]
        
        return return_
        
    def _prepare_reward_function(self):
        return_ = super()._prepare_reward_function()

        # get the body indices within the simulation (for estimating robot state)
        # if "penetrate_volume" in self.reward_names or "penetrate_depth" in self.reward_names:
        self._init_volume_sample_points()
        print("Total number of volume estimation points for each robot is:", self.volume_sample_points.shape[1])

        return return_

    def _init_volume_sample_points(self):
        """ Build sample points for penetration volume estimation
        NOTE: self.cfg.sim.body_measure_points must be a dict with
            key: body name (or part of the body name) to estimate
            value: dict(
                x, y, z: sample points to form a meshgrid
                transform: [x, y, z, roll, pitch, yaw] for transforming the meshgrid w.r.t body frame
            )
        """
        # read and specify the order of which body to sample from and its relative sample points.
        self.body_measure_name_order = [] # order specified
        self.body_sample_indices = []
        for idx in range(self.num_envs):
            rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[idx], self.actor_handles[idx])
            self.body_sample_indices.append([])
            for name, measure_name in itertools.product(rigid_body_names, self.cfg.sim.body_measure_points.keys()):
                if measure_name in name:
                    self.body_sample_indices[-1].append(
                        self.gym.find_actor_rigid_body_index(
                            self.envs[idx],
                            self.actor_handles[idx],
                            name,
                            gymapi.IndexDomain.DOMAIN_SIM,
                    ))
                    if idx == 0: # assuming all envs have the same actor configuration
                        self.body_measure_name_order.append(measure_name) # order specified
        self.body_sample_indices = torch.tensor(self.body_sample_indices, device= self.sim_device).flatten() # n_envs * num_bodies

        # compute and store each sample points in body frame.
        self.body_volume_points = dict()
        for measure_name, points_cfg in self.cfg.sim.body_measure_points.items():
            x = torch.tensor(points_cfg["x"], device= self.device, dtype= torch.float32, requires_grad= False)
            y = torch.tensor(points_cfg["y"], device= self.device, dtype= torch.float32, requires_grad= False)
            z = torch.tensor(points_cfg["z"], device= self.device, dtype= torch.float32, requires_grad= False)
            t = torch.tensor(points_cfg["transform"][0:3], device= self.device, dtype= torch.float32, requires_grad= False)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid_points = torch.stack([
                grid_x.flatten(),
                grid_y.flatten(),
                grid_z.flatten(),
            ], dim= -1) # n_points, 3
            q = quat_from_euler_xyz(
                torch.tensor(points_cfg["transform"][3], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][4], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][5], device= self.sim_device, dtype= torch.float32),
            )
            self.body_volume_points[measure_name] = tf_apply(
                q.expand(grid_points.shape[0], 4),
                t.expand(grid_points.shape[0], 3),
                grid_points,
            )
        num_sample_points_per_env = 0
        for body_name in self.body_measure_name_order:
            for measure_name in self.body_volume_points.keys():
                if measure_name in body_name:
                    num_sample_points_per_env += self.body_volume_points[measure_name].shape[0]
        self.volume_sample_points = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )
        self.velocity_sample_points = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )

    def _get_proprioception_obs(self, privileged= False):
        obs_buf = super()._get_proprioception_obs(privileged= privileged)
        
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False) and (not privileged):
            assert hasattr(self, "gravity_bias")
            proprioception_slice = get_obs_slice(self.obs_segments, "proprioception")
            obs_buf[:, proprioception_slice[0].start + 6: proprioception_slice[0].start + 9] += self.gravity_bias
        
        return obs_buf

    def _get_engaging_block_obs(self, privileged= False):
        """ Compute the obstacle info for the robot """
        if not self.check_BarrierTrack_terrain():
            # This could be wrong, check BarrierTrack implementation to get the exact shape.
            return torch.zeros((self.num_envs, (1 + (4 + 1) + 2)), device= self.sim_device)
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        self.refresh_volume_sample_points()
        return self.terrain.get_engaging_block_info(
            base_positions,
            self.volume_sample_points - base_positions.unsqueeze(-2), # (n_envs, n_points, 3)
        )

    def _get_sidewall_distance_obs(self, privileged= False):
        if not self.check_BarrierTrack_terrain():
            return torch.zeros((self.num_envs, 2), device= self.sim_device)
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        return self.terrain.get_sidewall_distance(base_positions)

    def _write_engaging_block_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "engaging_block"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.engaging_block * self.cfg.noise.noise_level * self.obs_scales.engaging_block
    
    def _write_sidewall_distance_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "sidewall_distance"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.sidewall_distance * self.cfg.noise.noise_level * self.obs_scales.sidewall_distance

    ##### adds-on with building the environment #####
    def _create_envs(self):        
        return_ = super()._create_envs()
        
        front_hip_names = getattr(self.cfg.asset, "front_hip_names", ["FR_hip_joint", "FL_hip_joint"])
        self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_hip_names):
            self.front_hip_indices[i] = self.dof_names.index(name)

        rear_hip_names = getattr(self.cfg.asset, "rear_hip_names", ["RR_hip_joint", "RL_hip_joint"])
        self.rear_hip_indices = torch.zeros(len(rear_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_hip_names):
            self.rear_hip_indices[i] = self.dof_names.index(name)

        hip_names = front_hip_names + rear_hip_names
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
            
        front_thigh_names = getattr(self.cfg.asset, "front_thigh_names", ["FR_thigh_joint", "FL_thigh_joint"])
        self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_hip_names):
            self.front_hip_indices[i] = self.dof_names.index(name)

        rear_thigh_names = getattr(self.cfg.asset, "rear_thigh_names", ["RR_thigh_joint", "RL_thigh_joint"])
        self.rear_thigh_indices = torch.zeros(len(rear_thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_thigh_names):
            self.rear_thigh_indices[i] = self.dof_names.index(name)
            
        rear_calf_names = getattr(self.cfg.asset, "rear_calf_names", ["HR_Knee_joint", "HL_Knee_joint"])
        self.rear_calf_indices = torch.zeros(len(rear_calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_calf_names):
            self.rear_calf_indices[i] = self.dof_names.index(name)

        thigh_names = front_thigh_names + rear_thigh_names
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
            
        front_dof_names = getattr(self.cfg.asset, "front_dof_names", ["FR_thigh_joint", "FL_thigh_joint", "FR_hip_joint", "FL_hip_joint", "FR_calf_joint", "FL_calf_joint"])
        self.front_dof_indices = torch.zeros(len(front_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_dof_names):
            self.front_dof_indices[i] = self.dof_names.index(name)
            
        rear_left_dof_names = getattr(self.cfg.asset, "rear_left_dof_names",["RL_thigh_joint", "RL_calf_joint"])
        self.rear_left_dof_indices = torch.zeros(len(rear_left_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_left_dof_names):
            self.rear_left_dof_indices[i] = self.dof_names.index(name)

        rear_right_dof_names = getattr(self.cfg.asset, "rear_right_dof_names",["RR_thigh_joint", "RR_calf_joint"])
        self.rear_right_dof_indices = torch.zeros(len(rear_right_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_right_dof_names):
            self.rear_right_dof_indices[i] = self.dof_names.index(name)
        
        return return_

    def _draw_volume_sample_points_vis(self):
        self.refresh_volume_sample_points()
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(1, 0.1, 0))
        sphere_penetrate_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(0.1, 0.6, 0.6))
        if self.cfg.terrain.selected == "BarrierTrack":
            penetration_mask = self.terrain.get_penetration_mask(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        for env_idx in range(self.num_envs):
            for point_idx in range(self.volume_sample_points.shape[1]):
                sphere_pose = gymapi.Transform(gymapi.Vec3(*self.volume_sample_points[env_idx, point_idx]), r= None)
                # if penetration_mask[env_idx, point_idx]:
                #     gymutil.draw_lines(sphere_penetrate_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose)
                # else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose)

    def _draw_debug_vis(self):
        return_ = super()._draw_debug_vis()

        if self.cfg.terrain.selected == "BarrierTrack":
            self.terrain.draw_virtual_terrain(self.viewer)
        if hasattr(self, "volume_sample_points") and self.cfg.viewer.draw_volume_sample_points:
            self._draw_volume_sample_points_vis()
        
        return return_

    ##### defines observation segments, which tells the order of the entire flattened obs #####
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = super().get_obs_segment_from_components(components)
    
        if "engaging_block" in components:
            if not self.check_BarrierTrack_terrain():
                # This could be wrong, please check the implementation of BarrierTrack
                segments["engaging_block"] = (1 + (4 + 1) + 2,)
            else:
                segments["engaging_block"] = get_terrain_cls("BarrierTrack").get_engaging_block_info_shape(self.cfg.terrain)
        if "sidewall_distance" in components:
            self.check_BarrierTrack_terrain()
            segments["sidewall_distance"] = (2,)
        
        return segments

    def refresh_volume_sample_points(self):
        if self.volume_sample_points_refreshed:
            return
        sampled_body_pos = self.all_rigid_body_states[self.body_sample_indices, :3].view(self.num_envs, -1, 3)
        sampled_body_quat = self.all_rigid_body_states[self.body_sample_indices, 3:7].view(self.num_envs, -1, 4)
        sample_points_start_idx = 0
        for body_idx, body_measure_name in enumerate(self.body_measure_name_order):
            num_volume_points = self.body_volume_points[body_measure_name].shape[0]                
            point_positions = tf_apply(
                    sampled_body_quat[:, body_idx].unsqueeze(1).expand(-1, num_volume_points, -1),
                    sampled_body_pos[:, body_idx].unsqueeze(1).expand(-1, num_volume_points, -1),
                    self.body_volume_points[body_measure_name].unsqueeze(0).expand(self.num_envs, -1, -1),
                ) # (num_envs, num_volume_points, 3)
            valid_velocity_mask = self.episode_length_buf > 0
            self.velocity_sample_points[valid_velocity_mask, sample_points_start_idx: sample_points_start_idx + num_volume_points] = \
                (point_positions[valid_velocity_mask] - self.volume_sample_points[valid_velocity_mask, sample_points_start_idx: sample_points_start_idx + num_volume_points]) / self.dt
            self.volume_sample_points[:, sample_points_start_idx: sample_points_start_idx + num_volume_points] = point_positions
            sample_points_start_idx += num_volume_points
        self.volume_sample_points_refreshed = True

    ##### Additional rewards #####
    def _reward_lin_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.base_lin_vel[:, :2]), dim= 1)

    def _reward_world_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)

    def _reward_tracking_world_vel(self):
        world_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)
        return torch.exp(-world_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_legs_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_legs_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(self.substep_torques * self.substep_dof_vel), dim=-1), dim=-1)

    def _reward_legs_energy_abs(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_alive(self):
        return 1.

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_lin_cmd(self):
        """ This reward term does not depend on the policy, depends on the command """
        return torch.norm(self.commands[:, :2], dim= 1)

    def _reward_lin_vel_x(self):
        return self.root_states[:, 7]
    
    def _reward_lin_vel_y_abs(self):
        return torch.abs(self.root_states[:, 8])
    
    def _reward_lin_vel_y_square(self):
        return torch.square(self.root_states[:, 8])
    
    def _reward_lin_vel_z_square(self):
        return torch.square(self.root_states[:, 9])

    def _reward_lin_pos_y(self):
        return torch.abs((self.root_states[:, :3] - self.env_origins)[:, 1])
    
    def _reward_yaw_abs(self):
        """ Aiming for the robot yaw to be zero (pointing to the positive x-axis) """
        yaw = get_euler_xyz(self.root_states[:, 3:7])[2]
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw < -np.pi] += np.pi * 2 # to range (-pi, pi)
        return torch.abs(yaw)

    def _reward_penetrate_depth(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_depths = self.terrain.get_penetration_depths(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_depths *= torch.norm(self.velocity_sample_points, dim= -1) + 1e-3
        return torch.sum(penetration_depths, dim= -1)

    def _reward_penetrate_volume(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_mask = self.terrain.get_penetration_mask(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_mask *= torch.norm(self.velocity_sample_points, dim= -1) + 1e-3
        return torch.sum(penetration_mask, dim= -1)

    def _reward_tilt_cond(self):
        """ Conditioned reward term in terms of whether the robot is engaging the tilt obstacle
        Use positive factor to enable rolling angle when incountering tilt obstacle
        """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        roll[roll > pi] -= pi * 2 # to range (-pi, pi)
        roll[roll < -pi] += pi * 2 # to range (-pi, pi)
        if hasattr(self, "volume_sample_points"):
            self.refresh_volume_sample_points()
            stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.volume_sample_points.view(-1, 3))
        else:
            stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.root_states[:, :3])
        stepping_obstacle_info = stepping_obstacle_info.view(self.num_envs, -1, stepping_obstacle_info.shape[-1])
        # Assuming that each robot will only be in one obstacle or non obstacle.
        robot_stepping_obstacle_id = torch.max(stepping_obstacle_info[:, :, 0], dim= -1)[0]
        tilting_mask = robot_stepping_obstacle_id == self.terrain.track_options_id_dict["tilt"]
        return_ = torch.where(tilting_mask, torch.clip(torch.abs(roll), 0, torch.pi/2), -torch.clip(torch.abs(roll), 0, torch.pi/2))
        return return_

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_front_hip_pos(self):
        """ Reward the robot to stop moving its front hips """
        return torch.sum(torch.square(self.dof_pos[:, self.front_hip_indices] - self.default_dof_pos[:, self.front_hip_indices]), dim=1)

    def _reward_rear_hip_pos(self):
        """ Reward the robot to stop moving its rear hips """
        return torch.sum(torch.square(self.dof_pos[:, self.rear_hip_indices] - self.default_dof_pos[:, self.rear_hip_indices]), dim=1)
    
    def _reward_rear_calf_pos(self):
        """ Reward the robot to stop moving its rear calfs """
        return torch.sum(torch.square(self.dof_pos[:, self.rear_calf_indices] - self.default_dof_pos[:, self.rear_calf_indices]), dim=1)
    
    def _reward_rear_pos_balance(self):
        """ Reward the robot to stop moving its rear hips """
        reward = torch.sum(torch.square(self.dof_pos[:, self.rear_left_dof_indices] - self.dof_pos[:, self.rear_right_dof_indices]), dim=1)
        return reward
    
    def _reward_front_dof_pos(self):
        """ Reward the robot to stop moving its front dofs """
        reward = torch.sum(torch.square(self.dof_pos[:, self.front_dof_indices] - self.default_dof_pos[:, self.front_dof_indices]), dim=1)
        reward *= (self.episode_length_buf > self.cfg.rewards.allow_contact_steps).float()
        return reward
    
    def _reward_front_dof_vel(self):
        # Penalize front dof velocities
        reward = torch.sum(torch.square(self.dof_vel[:, self.front_dof_indices]), dim=1)
        reward *= (self.episode_length_buf > self.cfg.rewards.allow_contact_steps).float()
        return reward
    
    def _reward_front_dof_acc(self):
        # Penalize front dof accelerations
        reward = torch.sum(torch.square((self.last_dof_vel[:, self.front_dof_indices] - self.dof_vel[:, self.front_dof_indices]) / self.dt), dim=1)
        reward *= (self.episode_length_buf > self.cfg.rewards.allow_contact_steps).float()
        return reward
    
    def _reward_thigh_acc(self):
        """ Reward the robot to stop moving its rear thighs """
        return torch.sum(torch.square((self.last_dof_vel[:, self.thigh_indices] - self.dof_vel[:, self.thigh_indices]) / self.dt), dim=1)
    
    def _reward_down_cond(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        pitch[pitch > pi] -= pi * 2 # to range (-pi, pi)
        pitch[pitch < -pi] += pi * 2 # to range (-pi, pi)
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] < 0.)
        pitch_err = torch.abs(pitch - 0.2)
        return torch.exp(-pitch_err/self.cfg.rewards.tracking_sigma) * engaging_mask # the higher positive factor, the more you want the robot to pitch down 0.2 rad

    def _reward_jump_x_vel_cond(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] > 0.) \
            & (engaging_obstacle_info[:, 0] > 0) # engaging jump-up, not engaging jump-down, positive distance.
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        pitch[pitch > pi] -= pi * 2 # to range (-pi, pi)
        pitch[pitch < -pi] += pi * 2 # to range (-pi, pi)
        pitch_up_mask = pitch < -0.75 # a hack value

        return torch.clip(self.base_lin_vel[:, 0], max= 1.5) * engaging_mask * pitch_up_mask

    def _reward_sync_legs_cond(self):
        """ A hack to force same actuation on both rear legs when jump. """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] > 0.) \
            & (engaging_obstacle_info[:, 0] > 0) # engaging jump-up, not engaging jump-down, positive distance.
        rr_legs = torch.clone(self.actions[:, 6:9]) # shoulder, thigh, calf
        rl_legs = torch.clone(self.actions[:, 9:12]) # shoulder, thigh, calf
        rl_legs[:, 0] *= -1 # flip the sign of shoulder action
        return torch.norm(rr_legs - rl_legs, dim= -1) * engaging_mask
    
    def _reward_sync_all_legs_cond(self):
        """ A hack to force same actuation on both front/rear legs when jump. """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] > 0.) \
            & (engaging_obstacle_info[:, 0] > 0) # engaging jump-up, not engaging jump-down, positive distance.
        right_legs = torch.clone(torch.cat([
            self.actions[:, 0:3],
            self.actions[:, 6:9],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs = torch.clone(torch.cat([
            self.actions[:, 3:6],
            self.actions[:, 9:12],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs[:, 0] *= -1 # flip the sign of shoulder action
        left_legs[:, 3] *= -1 # flip the sign of shoulder action
        return torch.norm(right_legs - left_legs, p= 1, dim= -1) * engaging_mask
    
    def _reward_sync_all_legs(self):
        right_legs = torch.clone(torch.cat([
            self.actions[:, 0:3],
            self.actions[:, 6:9],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs = torch.clone(torch.cat([
            self.actions[:, 3:6],
            self.actions[:, 9:12],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs[:, 0] *= -1 # flip the sign of shoulder action
        left_legs[:, 3] *= -1 # flip the sign of shoulder action
        return torch.norm(right_legs - left_legs, p= 1, dim= -1)
    
    def _reward_dof_error_cond(self):
        """ Force dof error when not engaging obstacle """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1] > 0)
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1) * engaging_mask
        
    def _reward_leap_bonous_cond(self):
        """ counteract the tracking reward loss during leap"""
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["leap"]] > 0) \
            & (-engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 1] < engaging_obstacle_info[:, 0]) \
            & (engaging_obstacle_info[:, 0] < 0.) # engaging jump-up, not engaging jump-down, positive distance.

        world_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)
        return (1 - torch.exp(-world_vel_error/self.cfg.rewards.tracking_sigma)) * engaging_mask # reverse version of tracking reward
    

    @torch.no_grad()
    def _reward_intrinsic_rnd(self):
        """Compute ‖f(s)-f̂(s)‖₂ (no grad, no scale)."""
        x = torch.cat([self.dof_pos, self.dof_vel], dim=-1)
        tgt  = self.rnd_target(x)
        pred = self.rnd_predictor(x)
        return torch.norm(tgt - pred, dim=-1)           # (num_envs,)

    def train_rnd_from_obs(self, input: torch.Tensor) -> float:
        """One SGD step on predictor using a batch of obs; return loss (float)."""
        with torch.no_grad():
            tgt = self.rnd_target(input)
        pred = self.rnd_predictor(input)
        loss = F.mse_loss(pred, tgt)
        self.rnd_opt.zero_grad()
        loss.backward()
        self.rnd_opt.step()
        return loss.item()