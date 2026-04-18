#!/usr/bin/env python3
"""Play script for diffusion-based locomotion policy."""

import os
import time
import json
import numpy as np
import isaacgym
from isaacgym import gymtorch, gymapi
import torch
import re
import random

from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, Logger
from legged_gym.utils.observation import get_obs_slice
from legged_gym.debugger import break_into_debugger
from legged_gym.utils.helpers import update_class_from_dict

from rsl_rl.modules.diffusion import GaussianDiffusion, ValueDiffusion
from rsl_rl.modules.temporal import TemporalUnet, ValueFunction
from rsl_rl.algorithms.diffuser import GuidedPolicy, ValueGuide, PerTransitionRewardModel
import rsl_rl.modules.hand_crafted_rewards as AnalyticReward
import rsl_rl.modules.constraints as Constraint
from rsl_rl.utils.cfg_helpers import instantiate_from_cfg


def create_recording_camera(gym, env_handle,
        resolution=(1920, 1080),
        h_fov=86,
        actor_to_attach=None,
        transform=None,
    ):
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_to_attach,
            transform,
            gymapi.FOLLOW_POSITION,
        )
    elif transform is not None:
        gym.set_camera_transform(
            camera_handle,
            env_handle,
            transform,
        )
    return camera_handle

class DiffuserPlayRunner:
    def __init__(self, env, cfg, device='cpu', load_path=None):
        """
        Builds the diffuser components based on the configuration.
        Assumes cfg supports dot access (e.g. via a DotDict or OmegaConf).
        
        Parameters:
            env: the environment instance.
            cfg: configuration object (with dot-access) that includes policy and runner settings.
            device: computation device (default 'cpu').
            load_path (str, optional): Path to a saved model state; if provided, the model state is loaded.
        """
        self.env = env
        self.cfg = cfg
        self.device = device
        self.observation_dim = self.cfg.policy.observation_dim
        self.action_dim = self.env.num_actions
        self.horizon = self.cfg.policy.horizon
        self.num_history = self.cfg.policy.num_history
        print("self.horizon", self.horizon)
        self.next_plan = None
        self.replan_margin = self.cfg.policy.replan_margin
        print("self.replan_margin", self.replan_margin)
        self.next_plan_start = 0

        self.obs_history = None

        self.diffuser_model = TemporalUnet(
            horizon=self.cfg.policy.horizon,
            transition_dim=self.cfg.policy.transition_dim,
            cond_dim=self.cfg.policy.cond_dim,
            dim=self.cfg.policy.dim,
            dim_mults=self.cfg.policy.dim_mults,
            attention=self.cfg.policy.attention,
        ).to(self.device)

        self.diffuser = GaussianDiffusion(
            self.diffuser_model,
            horizon=self.horizon,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_timesteps=self.cfg.runner.n_timesteps,
            loss_type=self.cfg.runner.loss_type,
            clip_denoised=self.cfg.runner.clip_denoised,
            predict_epsilon=self.cfg.runner.predict_epsilon,
            action_weight=self.cfg.runner.action_weight,
            loss_discount=self.cfg.runner.loss_discount,
            loss_weights=self.cfg.runner.loss_weights,
            num_history=self.cfg.policy.num_history,
            execute_num=self.cfg.policy.execute_num,
            variance_clip = self.cfg.policy.variance_clip,
        )
        ecfg = self.cfg.policy.early_cache
        enable = ecfg["enable"]
        m_steps = ecfg["m_steps"]
        self.diffuser.set_early_cache(enable=enable, m_steps=m_steps)
        self._cache_reset_thresh = ecfg["reset_threshold"]
        self._cache_reset_type = ecfg["reset_type"]
        self._denoise_calls = 0
        self._denoise_steps_sum = 0
        self._skip_m = int(m_steps) if enable else 0
        self._cache_ready = False
        
        self.diffuser.enable_telemetry(
            out_dir="logs/exp_diff/telemetry",
            flush_every_rows=500,
            tag="play"
        )

        
        self.reward_model = ValueDiffusion(
            model=ValueFunction(
                horizon=self.horizon,
                transition_dim=self.observation_dim + self.action_dim,
                cond_dim=self.observation_dim,
                dim=self.cfg.reward_model.full.dim,
                dim_mults=self.cfg.reward_model.full.dim_mults,
                out_dim=1,
            ),
            horizon=self.horizon,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_timesteps=self.cfg.runner.n_timesteps,
            loss_type=self.cfg.runner.value_loss_type,
            clip_denoised=self.cfg.runner.clip_denoised,
            predict_epsilon=self.cfg.runner.predict_epsilon,
            action_weight=self.cfg.runner.action_weight,
            loss_discount=self.cfg.runner.loss_discount,
            loss_weights=self.cfg.runner.loss_weights,
        ).to(self.device)
        
        if load_path is not None:
            self.load(load_path, load_optimizer=True)

        gp = self.cfg.guided_policy

        assert hasattr(gp, "reward_type"), "cfg.guided_policy.reward_type is required ('nn' or 'analytic')."
        if gp.reward_type == "analytic":
            assert hasattr(gp, "reward_name"), "cfg.guided_policy.reward_name is required for analytic reward."
            assert hasattr(gp, "reward_args"), "cfg.guided_policy.reward_args is required for analytic reward."
            self.reward_model = instantiate_from_cfg(
                module=AnalyticReward,
                class_name=gp.reward_name,
                raw_args=gp.reward_args,
                device=self.device,
            )

        elif gp.reward_type == "nn":
            pass
        else:
            raise AssertionError(f"Unsupported cfg.guided_policy.reward_type={gp.reward_type}")

        self.guide = ValueGuide(self.reward_model)
        self.guided_policy = GuidedPolicy(
            guide=self.guide,
            diffusion_model=self.diffuser,
            **self.cfg.guided_policy.sample_kwargs
        )
        self.guided_policy.scale = torch.full(
            (self.env.num_envs,),
            self.cfg.guided_policy.sample_kwargs["scale"],
            dtype=torch.float32,
            device=self.device
        )

        assert hasattr(gp, "candidate_same_as_reward"), "cfg.guided_policy.candidate_same_as_reward is required (bool)."
        if gp.candidate_same_as_reward:
            self.candidate_reward_model = self.reward_model
        else:
            assert hasattr(gp, "candidate_type"), "cfg.guided_policy.candidate_type is required when candidate_same_as_reward=False."
            if gp.candidate_type == "analytic":
                assert hasattr(gp, "candidate_name"), "cfg.guided_policy.candidate_name is required for analytic candidate."
                assert hasattr(gp, "candidate_args"), "cfg.guided_policy.candidate_args is required for analytic candidate."
                self.candidate_reward_model = instantiate_from_cfg(
                    module=AnalyticReward,
                    class_name=gp.candidate_name,
                    raw_args=gp.candidate_args,
                    device=self.device,
                )
            elif gp.candidate_type == "nn":
                assert hasattr(gp, "candidate_path"), "cfg.guided_policy.candidate_path is required for nn candidate."
                self.candidate_reward_model = ValueDiffusion(
                    model=ValueFunction(
                        horizon=self.horizon,
                        transition_dim=self.observation_dim + self.action_dim,
                        cond_dim=self.observation_dim,
                        dim=self.cfg.reward_model.full.dim,
                        dim_mults=self.cfg.reward_model.full.dim_mults,
                        out_dim=1,
                    ),
                    horizon=self.horizon,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    n_timesteps=self.cfg.runner.n_timesteps,
                    loss_type=self.cfg.runner.value_loss_type,
                    clip_denoised=self.cfg.runner.clip_denoised,
                    predict_epsilon=self.cfg.runner.predict_epsilon,
                    action_weight=self.cfg.runner.action_weight,
                    loss_discount=self.cfg.runner.loss_discount,
                    loss_weights=self.cfg.runner.loss_weights,
                ).to(self.device)

                cand_ckpt = torch.load(gp.candidate_path, map_location=self.device)
                self.candidate_reward_model.load_state_dict(cand_ckpt["reward_model_state_dict"], strict=True)
            else:
                raise AssertionError(f"Unsupported cfg.guided_policy.candidate_type={gp.candidate_type}")

        assert hasattr(gp, "apply_constraint"), "cfg.guided_policy.apply_constraint is required (bool)."
        chain = Constraint.ConstraintChain()
        if gp.apply_constraint:
            names = gp.constraint_name
            args  = gp.constraint_args

            if isinstance(names, (list, tuple)):
                assert isinstance(args, (list, tuple)) and len(args) == len(names), \
                    "constraint_name/constraint_args must be lists of the same length"
                for cname, cargs in zip(names, args):
                    c_obj = instantiate_from_cfg(Constraint, cname, cargs, self.device)
                    chain.add(c_obj)
            else:
                c_obj = instantiate_from_cfg(Constraint, names, args, self.device)
                chain.add(c_obj)
        else:
            pass

        self.guided_policy.sample_kwargs["constraints"] = chain

        self.execute_num = self.cfg.policy.execute_num
        assert self.execute_num > 2 * self.replan_margin, \
            f"execute_num ({self.execute_num}) must be greater than 2 * replan_margin ({self.replan_margin})"
        
        self.planned_actions = None
        self.current_plan_index = 0
        
        self.diff_obs_scale = torch.tensor(self.cfg.policy.diff_obs_scale, device=self.device)

        self.env.reset()

    def play_step(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return one action for each environment for this simulator step.

        Re-planning strategy
        --------------------
        * Let `execute_num` be the length of each action chunk.
        * While executing the current chunk (`planned_actions`), once the
        *remaining* steps equal `replan_margin`, we pre-compute a new
        action chunk (`next_plan`) **without interrupting** execution.
        * When the current chunk is exhausted, we switch to `next_plan`
        and start from index `replan_margin` (because those first
        `replan_margin` actions conceptually overlap with the tail of
        the previous chunk that was just executed).
        """

        num_envs = obs.shape[0]

        obs = obs * self.diff_obs_scale
        if self.obs_history is None:
            self.obs_history = obs.unsqueeze(1).repeat(1, self.num_history, 1)
        else:
            self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
            self.obs_history[:, -1, :] = obs

        conditions = {i: self.obs_history[:, i, :] for i in range(self.num_history)}
        if hasattr(self, 'diffuser') and hasattr(self.diffuser, 'invalidate_cache'):
            last_obs = self.obs_history[:, -1, :]
            if not hasattr(self, '_prev_last_obs') or self._prev_last_obs is None:
                self._prev_last_obs = last_obs.clone()
            else:
                if getattr(self, '_cache_reset_type', 'all') == 'cmd':
                    diff = (last_obs[:, 9:12] - self._prev_last_obs[:, 9:12]).abs().sum(dim=-1).mean()
                else:
                    diff = (last_obs - self._prev_last_obs).norm(dim=-1).mean()
                if diff.item() > getattr(self, '_cache_reset_thresh', float('inf')):
                    self.diffuser.invalidate_cache()
                    self._prev_last_obs = last_obs.clone()
                    self._cache_ready = False

        remaining = self.execute_num - self.current_plan_index
        if remaining <= self.replan_margin and self.next_plan is None:
            print("replanning with remaining steps", remaining)
            T = int(self.diffuser.n_timesteps)
            steps = (T - self._skip_m) if (self._cache_ready and self._skip_m > 0) else T
            self._denoise_steps_sum += steps; self._denoise_calls += 1
            if steps == T: self._cache_ready = True
            next_plan, _ = self.guided_policy(
                conditions,
                batch_size=num_envs,
                num_candidate=self.cfg.policy.num_candidate,
                verbose=False,
                candidate_reward_model=self.candidate_reward_model
            )

            noise_scale = self.cfg.policy.action_noise_scale
            if noise_scale is not None:
                next_plan = next_plan + torch.randn_like(next_plan) * noise_scale

            scale_tensor = torch.tensor(
                self.cfg.policy.diff_action_scale,
                device=self.device,
                dtype=next_plan.dtype
            )
            self.next_plan = next_plan / scale_tensor
            self.next_plan_start = self.replan_margin

        if self.planned_actions is None or self.current_plan_index >= self.execute_num:
            print("Use Plan Till Index:", self.current_plan_index)
            if self.next_plan is not None:
                self.planned_actions = self.next_plan
                self.current_plan_index = self.next_plan_start
                print("Switched to next plan with start index", self.current_plan_index)
                self.next_plan = None
            else:
                T = int(self.diffuser.n_timesteps)
                steps = (T - self._skip_m) if (self._cache_ready and self._skip_m > 0) else T
                self._denoise_steps_sum += steps; self._denoise_calls += 1
                if steps == T: 
                    self._cache_ready = True
                print("self.cfg.policy.num_candidate", self.cfg.policy.num_candidate)
                self.planned_actions, _ = self.guided_policy(
                    conditions,
                    batch_size=num_envs,
                    num_candidate=self.cfg.policy.num_candidate,
                    verbose=False,
                    candidate_reward_model=self.candidate_reward_model
                )
                noise_scale = self.cfg.policy.action_noise_scale
                if noise_scale is not None:
                    self.planned_actions += torch.randn_like(self.planned_actions) * noise_scale
                scale_tensor = torch.tensor(
                    self.cfg.policy.diff_action_scale,
                    device=self.device,
                    dtype=self.planned_actions.dtype
                )
                self.planned_actions = self.planned_actions / scale_tensor
                self.current_plan_index = 0

        action = self.planned_actions[:, self.current_plan_index, :]
        self.current_plan_index += 1
        return action.to(self.device)
    
    def load(self, path, load_optimizer=True):
        """Load model state from checkpoint."""
        print("Loading model from", path)
        loaded_dict = torch.load(path, map_location=self.device)
        self.diffuser.load_state_dict(loaded_dict['diffuser_state_dict'], strict=False)
        gp = self.cfg.guided_policy
        if hasattr(gp, "reward_load_external") and gp.reward_load_external:
            assert hasattr(gp, "reward_external_path"), \
                "cfg.guided_policy.reward_external_path must be provided when reward_load_external=True."
            ext_ckpt = torch.load(gp.reward_external_path, map_location=self.device)
            self.reward_model.load_state_dict(ext_ckpt['reward_model_state_dict'], strict=True)
            print(f"[load] reward_model loaded from external: {gp.reward_external_path}")
        else:
            self.reward_model.load_state_dict(loaded_dict['reward_model_state_dict'], strict=False)
        
        self.current_learning_iteration = loaded_dict['iter']
        
        print(f"Loaded model from {path} at iteration {self.current_learning_iteration}")
        return loaded_dict.get('infos', None)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    train_cfg.runner.resume = False # Just for testing random policy
    env_cfg.seed = 1
    
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.num_rows = 2
    env_cfg.init_state.pos = [0, 0, 0.35]
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.unify = False
    env_cfg.terrain.max_Zscale = 0.0
    env_cfg.domain_rand.init_base_vel_range = [0., 0.]
    env_cfg.domain_rand.init_base_rot_range = dict(roll=[0, 0], pitch=[-0., -0.], yaw=[0, 0])
    env_cfg.viewer.debug_viz = True
    env_cfg.viewer.draw_volume_sample_points = False
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    env_cfg.commands.resampling_time = int(1e16)
    env_cfg.domain_rand.friction_range = [0, 2]
    env_cfg.env.use_lin_vel = False
    env_cfg.env.use_friction = False
    train_cfg.guided_policy.resample_guidance_scale = False
    K_FRAMES = 100        # switch after K steps since env 0 reset
    NEW_SCALE = 10.0      # target guidance scale
    SWITCH_SCALE = False

    if env_cfg.env.num_envs > 1:
        env_cfg.viewer.pos = [4., 5.4, 0.2]
        env_cfg.viewer.lookat = [4., 3.8, 0.2]
    else:
        env_cfg.viewer.pos = [4., -6., 1]
        env_cfg.viewer.lookat = [5., -7., 0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("Environment created.")
    env.reset()

    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_P, "push_robot")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_T, "press_robot")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_J, "action_jitter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_Q, "exit")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_R, "agent_full_reset")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_U, "full_reset")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_C, "resample_commands")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_W, "forward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_S, "backward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_A, "leftward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_D, "rightward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_F, "leftturn")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_G, "rightturn")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_B, "leftdrag")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_M, "rightdrag")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_X, "stop")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_K, "mark")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_N, "env_left")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_M, "env_right")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_O, "env_up")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_L, "env_down")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_B, "curriculum_reset")

    logger = Logger(env.dt)
    robot_index = 0
    stop_state_log = args.plot_time
    stop_rew_log = env.max_episode_length + 1
    start_time = time.time_ns()

    if RECORD_FRAMES:
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*env_cfg.viewer.pos)
        transform.r = gymapi.Quat.from_euler_zyx(0., 0., -np.pi/2)
        recording_camera = create_recording_camera(env.gym, env.envs[0], transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_log_dir = os.path.join("logs", train_cfg.runner.experiment_name)
    load_run_folder = args.load_run if args.load_run is not None else env_cfg.runner.load_run
    folder_path = os.path.join(base_log_dir, load_run_folder)

    if args.checkpoint is not None:
        pattern = re.compile(rf'_{re.escape(str(args.checkpoint))}\.pt$')
        matching_files = [file for file in os.listdir(folder_path) if pattern.search(file)]
        
        if not matching_files:
            raise FileNotFoundError(f"No checkpoint file ending with _{args.checkpoint}.pt found in the directory.")
        
        checkpoint_path = os.path.join(folder_path, matching_files[0])
    else:
        checkpoint_files = [
            file for file in os.listdir(folder_path)
            if re.search(r'_\d+\.pt$', file)
        ]

        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files matching the criteria found in the directory.")

        def extract_number(filename):
            """Extract the numeric part located between the underscore and the '.pt' extension in the file name."""
            match = re.search(r'_(\d+)\.pt$', filename)
            return int(match.group(1)) if match else -1

        latest_file = max(checkpoint_files, key=extract_number)
        checkpoint_path = os.path.join(folder_path, latest_file)

    print("Checkpoint path:", checkpoint_path)
    diffuser_runner = DiffuserPlayRunner(env, train_cfg, device=device, load_path=checkpoint_path)

    diffuser_runner._switched_this_episode = False
    img_idx = 0
    frame = 0
    reward_buffer = deque(maxlen=400)
    vel_buffer = deque(maxlen=400)
    
    env.commands[:, 0] = 0.0
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    while True:
        mark = 0.0  # for user marking timestep
        if args.slow > 0:
            time.sleep(args.slow)

        obs = env.get_observations()
        if frame % 10 == 0:
            vel = env.base_lin_vel
            vel_buffer.append(vel[..., 0] / 2)
            avg_vel = sum(vel_buffer) / len(vel_buffer)
            print(f"[Frame {frame:05d}] Avg velocity over last {len(vel_buffer)} frames: {avg_vel.item():.4f}")
        s_i = obs.detach()

        if not diffuser_runner._switched_this_episode and SWITCH_SCALE:
            steps0 = int(env.episode_length_buf[0].item())
            if steps0 >= K_FRAMES:
                mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                mask[0] = True
                diffuser_runner.guided_policy.resample_scale(mask, NEW_SCALE)
                diffuser_runner._switched_this_episode = True
                print(f"[Scale] env 0 guidance scale -> {NEW_SCALE} at step {steps0}")

        actions = diffuser_runner.play_step(obs.detach())

        if args.zero_act_until > 0:
            zero_mask = env.episode_length_buf < args.zero_act_until
            zero_act = torch.tensor(
                [0, 2.0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0],
                device=actions.device,
                dtype=actions.dtype,
            )
            actions[zero_mask] = zero_act
            env.root_states[env.episode_length_buf == args.zero_act_until, 7:10] = 0.
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if hasattr(diffuser_runner.guided_policy, "reset"):
                diffuser_runner.guided_policy.reset(zero_mask)

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if frame % 10 == 0:
            current_reward = rews[robot_index].item()  # get scalar reward
            print(f"[Frame {frame:05d}] current‑reward = {current_reward:.4f}")
            reward_buffer.append(current_reward)
            avg_reward = sum(reward_buffer) / len(reward_buffer)
            print(f"[Frame {frame:05d}] Avg reward over last {len(reward_buffer)} frames: {avg_reward:.4f}")

        if RECORD_FRAMES:
            filename = os.path.join(os.path.abspath("logs/images/"), f"{img_idx:04d}.png")
            print("Recording frame:", filename)
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        if MOVE_CAMERA:
            camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
            camera_position = env.root_states[0, :3].cpu().numpy() - camera_direction
            env.set_camera(camera_position, camera_position + camera_direction)

        for ui_event in env.gym.query_viewer_action_events(env.viewer):
            if ui_event.action == "push_robot" and ui_event.value > 0:
                env._push_robots()
            if ui_event.action == "press_robot" and ui_event.value > 0:
                env.root_states[:, 9] = torch_rand_float(-env.cfg.domain_rand.max_push_vel_xy, 0,
                                                          (env.num_envs, 1), device=env.device).squeeze(1)
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "action_jitter" and ui_event.value > 0:
                obs, critic_obs, rews, dones, infos = env.step(actions + torch.randn_like(actions) * 0.2)
            if ui_event.action == "exit" and ui_event.value > 0:
                print("Exiting diffuser play script.")
                exit(0)
            if ui_event.action == "agent_full_reset" and ui_event.value > 0:
                print("agent_full_reset")
                if hasattr(diffuser_runner.guided_policy, "reset"):
                    diffuser_runner.guided_policy.reset()
            if ui_event.action == "full_reset" and ui_event.value > 0:
                print("full_reset")
                if hasattr(diffuser_runner.guided_policy, "reset"):
                    diffuser_runner.guided_policy.reset()
                print(env._get_terrain_curriculum_move([robot_index]))
                obs, _ = env.reset()
                env.commands[:, 2] = 0
            if ui_event.action == "resample_commands" and ui_event.value > 0:
                print("resample_commands")
                env._resample_commands(torch.arange(env.num_envs, device=env.device))
            if ui_event.action == "stop" and ui_event.value > 0:
                env.commands[:, :3] = 0
            if ui_event.action == "forward" and ui_event.value > 0:
                env.commands[:, 0] = env_cfg.commands.ranges.lin_vel_x[1]
            if ui_event.action == "backward" and ui_event.value > 0:
                env.commands[:, 0] = env_cfg.commands.ranges.lin_vel_x[0]
            if ui_event.action == "leftward" and ui_event.value > 0:
                env.commands[:, 1] = env_cfg.commands.ranges.lin_vel_y[1]
            if ui_event.action == "rightward" and ui_event.value > 0:
                env.commands[:, 1] = env_cfg.commands.ranges.lin_vel_y[0]
            if ui_event.action == "leftturn" and ui_event.value > 0:
                env.commands[:, 2] = env_cfg.commands.ranges.ang_vel_yaw[1]
            if ui_event.action == "rightturn" and ui_event.value > 0:
                env.commands[:, 2] = env_cfg.commands.ranges.ang_vel_yaw[0]
            if ui_event.action == "leftdrag" and ui_event.value > 0:
                env.root_states[:, 7:10] += quat_rotate(env.base_quat,
                                              torch.tensor([[0., 0.5, 0.]], device=env.device))
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "rightdrag" and ui_event.value > 0:
                env.root_states[:, 7:10] += quat_rotate(env.base_quat,
                                              torch.tensor([[0., -0.5, 0.]], device=env.device))
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "mark" and ui_event.value > 0:
                mark = 0.05
            if ui_event.action == "env_left" and ui_event.value > 0:
                if env.terrain_types + 1 < env_cfg.terrain.num_cols:
                    env.terrain_types = env.terrain_types + 1
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            if ui_event.action == "env_right" and ui_event.value > 0:
                if env.terrain_types > 0:
                    env.terrain_types = env.terrain_types - 1
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            if ui_event.action == "env_up" and ui_event.value > 0:
                if env.terrain_levels + 1 < env.max_terrain_level:
                    env.terrain_levels = env.terrain_levels + 1
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            if ui_event.action == "env_down" and ui_event.value > 0:
                if env.terrain_levels > 0:
                    env.terrain_levels = env.terrain_levels - 1
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            if ui_event.action == "curriculum_reset" and ui_event.value > 0:
                env._resample_commands(torch.arange(env.num_envs, device=env.device))


        if dones.any():
            if hasattr(diffuser_runner.guided_policy, "reset"):
                diffuser_runner.guided_policy.reset(dones)

            diffuser_runner.obs_history = None

            diffuser_runner.planned_actions = None
            diffuser_runner.current_plan_index = 0

            diffuser_runner.next_plan = None
            diffuser_runner.next_plan_start = 0

            reward_buffer.clear()
            diffuser_runner.diffuser.invalidate_cache()
            diffuser_runner._prev_last_obs = None
            diffuser_runner._cache_ready = False

            if bool(dones[0].item()):
                mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                mask[0] = True
                diffuser_runner.guided_policy.resample_scale(mask)
                diffuser_runner._switched_this_episode = False


            obs, _ = env.reset()

        if frame % 100 == 0:
            avg_steps = (diffuser_runner._denoise_steps_sum / max(1, diffuser_runner._denoise_calls))
            T = diffuser_runner.diffuser.n_timesteps
            saved_pct = (1.0 - (avg_steps / T)) * 100.0
            print("frame_rate:", 100 / ((time.time_ns() - start_time) * 1e-9),
                  "command_x:", env.commands[robot_index, 0],
                  "root_state_x:", env.root_states[robot_index, 0],
                  f"avg_denoise_steps:{avg_steps:.3f}/{T}",
                  f"saving:{saved_pct:.2f}%")
            start_time = time.time_ns()

        frame += 1

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    
    args = get_args([
            dict(name= "--slow", type= float, default= 0., help= "slow down the simulation by sleep secs (float) every frame"),
            dict(name= "--zero_act_until", type= int, default= 0., help= "zero action until this step"),
            dict(name= "--sample", action= "store_true", default= False, help= "sample actions from policy"),
            dict(name= "--plot_time", type= int, default= 100, help= "plot states after this time"),
            dict(name= "--no_throw", action= "store_true", default= False),
            ])
    play(args)