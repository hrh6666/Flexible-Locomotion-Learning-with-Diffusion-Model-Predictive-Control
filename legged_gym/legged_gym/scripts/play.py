# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import OrderedDict
import os
import json
import time
import numpy as np
np.float = np.float32
import isaacgym
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import update_class_from_dict
from legged_gym.utils.observation import get_obs_slice
from legged_gym.debugger import break_into_debugger

import numpy as np
import torch
import torch.nn as nn

def create_recording_camera(gym, env_handle,
        resolution= (1920, 1080),
        h_fov= 86,
        actor_to_attach= None,
        transform= None, # related to actor_to_attach
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

def play(args):
    # break_into_debugger() # uncomment this to use vscode debugger
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # with open(os.path.join("logs", train_cfg.runner.experiment_name, args.load_run, "config.json"), "r") as f:
    #     d = json.load(f, object_pairs_hook= OrderedDict)
    #     update_class_from_dict(env_cfg, d, strict= True)
    #     update_class_from_dict(train_cfg, d, strict= True)
    # env_cfg.env.obs_components = env_cfg.env.privileged_obs_components
    # override some parameters for testing
    goal_command_override = True
    train_cfg.runner.resume = True

    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    # env_cfg.init_state.pos = [0, 0, 0.32]
    env_cfg.terrain.bipedal_cols = 3.


    env_cfg.commands.heading_command = False
    # env_cfg.control.action_scale = 0.5
    env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.friction_range = [2., 2.]
    env_cfg.terrain.unify=False
    env_cfg.terrain.max_Zscale = 0.0

    #333env_cfg.domain_rand.init_dof_pos_ratio_range = [0.5, 1.5]
    env_cfg.domain_rand.init_base_vel_range = [0., 0.]
    env_cfg.domain_rand.init_base_rot_range = dict(
            roll= [0, 0],
            pitch= [0, 0],
            yaw= [0,0]
        )
    env_cfg.viewer.debug_viz = True
    env_cfg.viewer.draw_volume_sample_points = False
    
    # env_cfg.commands.ranges.lin_vel_x=[-0.5 ,0.5]
    # env_cfg.commands.ranges.lin_vel_y=[-0.0,0.0]
    # env_cfg.commands.ranges.ang_vel_yaw=[-0.5,0.5]

    env_cfg.commands.ranges.lin_vel_x=[-1. , 1.0]
    env_cfg.commands.ranges.lin_vel_y=[-0.5,0.5]
    env_cfg.commands.ranges.ang_vel_yaw=[-1.0,1.0]
    
    env_cfg.commands.resampling_time=int(1e16)
    if not env_cfg.env.goal_command:
        env_cfg.env.episode_length_s=int(1e16)
    # env_cfg.env.episode_length_s=int(1e16)
    # env_cfg.domain_rand.init_base_pos_range = dict(
    #         x= [-0.8, 0.8],
    #         y= [-0.2, 0.2],
    #     )
    
    if args.no_throw:
        env_cfg.domain_rand.init_base_vel_range = [0., 0.]
    elif isinstance(env_cfg.domain_rand.init_base_vel_range, dict):
        print("init_base_vel_range 'x' value is set to:", env_cfg.domain_rand.init_base_vel_range["x"])
    else:
        print("init_base_vel_range not set, remains:", env_cfg.domain_rand.init_base_vel_range)
    # env_cfg.termination.termination_terms = []
    
    ################ left
    # env_cfg.viewer.pos = [4., 5.4, 0.2]
    # env_cfg.viewer.lookat = [4., 3.8, 0.2]
    ################ right
    # env_cfg.viewer.pos = [4.4, 2.2, 0.3]
    # env_cfg.viewer.lookat = [4.4, 3.2, 0.3]
    if env_cfg.env.num_envs > 1:
        ################ left
        env_cfg.viewer.pos = [4., 5.4, 0.2]
        env_cfg.viewer.lookat = [4., 3.8, 0.2]
    else:
        ################ back
        env_cfg.viewer.pos = [4., -6., 1]
        # env_cfg.viewer.pos = [0., 2., 5]
        env_cfg.viewer.lookat = [5., -7., 0]
    ################ front
    # env_cfg.viewer.pos = [4., 4., 0.4]
    # env_cfg.viewer.lookat = [3., 4., 0.34]
    ################ right-back
    # env_cfg.viewer.pos = [1.0, 1.8, 0.5]
    # env_cfg.viewer.lookat = [1.6, 2.6, 0.15]
    ################ back lookdown
    # env_cfg.viewer.pos = [2.9, 4., 0.9]
    # env_cfg.viewer.lookat = [3., 4., 0.25]
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("clip_actions_method:", getattr(env.cfg.normalization, "clip_actions_method", None))
    env.reset()
    print("terrain_levels:", env.terrain_levels.float().mean(), env.terrain_levels.float().max(), env.terrain_levels.float().min())
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    # print(obs)
    # print(critic_obs)
    # register debugging options to manually trigger disruption
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_P, "push_robot")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_L, "press_robot")
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
    # load policy
    print(train_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        save_cfg= False,
    )
    agent_model = ppo_runner.alg.actor_critic
    if hasattr(train_cfg,"estimator"):
        estimator_model = ppo_runner.alg.estimator
    policy = ppo_runner.alg.act_play
    if args.sample:
        policy = agent_model.act
    # get obs_slice to read the obs
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    if RECORD_FRAMES:
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*env_cfg.viewer.pos)
        transform.r = gymapi.Quat.from_euler_zyx(0., 0., -np.pi/2)
        recording_camera = create_recording_camera(
            env.gym,
            env.envs[0],
            transform= transform,
        )

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 0 # which joint is used for logging
    stop_state_log = args.plot_time # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.6, 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    camera_follow_id = 0 # only effective when CAMERA_FOLLOW
    img_idx = 0
    goal_command_mode=0
    action_list=[]
    dof_list=[]
    vel_list=[]

    if hasattr(env, "motor_strength"):
        print("motor_strength:", env.motor_strength[0, robot_index].cpu().numpy().tolist())
    print("torque_limits:", env.torque_limits)
    print("init_dof_vel_range:", getattr(env.cfg.domain_rand, "init_dof_vel_range", None))
    print("init_dof_pos_ratio_range:", getattr(env.cfg.domain_rand, "init_dof_pos_ratio_range", None))
    start_time = time.time_ns()
    
    use_goal_cmd = False
    
    gating_weight = 0
    
    for i in range(100000*int(env.max_episode_length)):
        mark = 0. # use for user marking timestep
        # print("dofs", env.dof_pos[0, :].cpu().numpy())
        if args.slow > 0:
            time.sleep(args.slow)
            # print(torch.where(obs_component[robot_index, 1:6] > 0), obs_component[robot_index, 6:])
            # print(obs_component[robot_index])
        # if env.substep_exceed_dof_pos_limits.any():
        #     exceed_idxs = env.substep_exceed_dof_pos_limits.nonzero(as_tuple=True)
        #     print("dof pos limits exceeded: {}, {:.2f}, {:.2f}-{:.2f}".format(
        #         exceed_idxs[-1].tolist(),
        #         env.dof_pos[0, exceed_idxs[-1][0]].tolist(),
        #         *env.dof_pos_limits[exceed_idxs[-1].tolist()[0]].tolist(),
        #     ))
        # with torch.inference_mode():
        if use_goal_cmd:
            if goal_command_mode==0:
                if goal_command_override:
                    obs[:,9]=0.0
                    obs[:,10]=0.0
                    obs[:,48]=0.0
            elif goal_command_mode==1:
                if goal_command_override:
                    obs[:,9]=2*env_cfg.normalization.obs_scales.lin_vel
                    obs[:,10]=-0.4
                    obs[:,48]=4
            elif goal_command_mode==2:
                if goal_command_override:
                    obs[:,9]=0*env_cfg.normalization.obs_scales.lin_vel
                    obs[:,10]=0.4*env_cfg.normalization.obs_scales.lin_vel
                    obs[:,48]=4
            elif goal_command_mode==3:
                if goal_command_override:
                    obs[:,9]=0.*env_cfg.normalization.obs_scales.lin_vel
                    obs[:,10]=-0.5*env_cfg.normalization.obs_scales.lin_vel
                    obs[:,48]=4
            elif goal_command_mode==4:
                if goal_command_override:
                    obs[:,9]=-0.5*env_cfg.normalization.obs_scales.lin_vel
                    obs[:,10]=0.
                    obs[:,48]=4

        # vx=0.85*np.cos(5*np.pi/10)
        # # vx=0.3
        # obs[:,9]=vx*env_cfg.normalization.obs_scales.lin_vel
        # obs[:,10]=-np.sqrt(0.85**2-vx**2)*env_cfg.normalization.obs_scales.lin_vel
        # # obs[:,10]=-0.4*env_cfg.normalization.obs_scales.lin_vel
        # obs[:,48]=3
        # print("current frame", i)
        # with open("error_log.txt", "a") as f:
        #     f.write("current frame " + str(i) + "\n")
        print("obs", obs)
        actions = policy(obs.detach())
        print("actions", actions)
        
        # action_list.append(actions * env.cfg.control.action_scale + env.default_dof_pos)
        # dof_list.append(env.qdof_pos.clone()

        # print(env.base_lin_vel[robot_index, 0].item(), env.base_lin_vel[robot_index, 1].item(), env.base_ang_vel[robot_index, 2].item())
        # if i>50 and i<100:
        #     vel_list.append([env.base_lin_vel[robot_index, 0].item(), env.base_lin_vel[robot_index, 1].item(), env.base_ang_vel[robot_index, 2].item()])
        # elif i==200:
        #     vel_list=np.array(vel_list)
        #     print(vel_list.mean(axis=0))
        #     exit()

        if args.zero_act_until > 0:
            actions[env.episode_length_buf < args.zero_act_until] = 0.
            env.root_states[env.episode_length_buf == args.zero_act_until, 7:10] = 0.
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if hasattr(ppo_runner.alg, "teacher_actor_critic"):
                ppo_runner.alg.teacher_actor_critic.reset(env.episode_length_buf == args.zero_act_until)
            agent_model.reset(env.episode_length_buf == args.zero_act_until)
        
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        # print(obs)
        # print(env.time_info)

        if RECORD_FRAMES:
            filename = os.path.join(
                os.path.abspath("logs/images/"),
                f"{img_idx:04d}.png",
            )
            print('filename:',filename)
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            # env.gym.render_all_camera_sensors(env.sim)
            # env.gym.write_camera_image_to_file(
            #     env.sim,
            #     env.envs[0],
            #     recording_camera,
            #     gymapi.IMAGE_COLOR,
            #     filename,
            # )
            img_idx += 1
        if MOVE_CAMERA:
            if CAMERA_FOLLOW:
                camera_position[:] = env.root_states[camera_follow_id, :3].cpu().numpy() - camera_direction
            else:
                camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        for ui_event in env.gym.query_viewer_action_events(env.viewer):
            if ui_event.action == "push_robot" and ui_event.value > 0:
                # manully trigger to push the robot
                env._push_robots()
            if ui_event.action == "press_robot" and ui_event.value > 0:
                env.root_states[:, 9] = torch_rand_float(-env.cfg.domain_rand.max_push_vel_xy, 0, (env.num_envs, 1), device=env.device).squeeze(1)
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "action_jitter" and ui_event.value > 0:
                # assuming wrong action is taken
                obs, critic_obs, rews, dones, infos = env.step(actions + torch.randn_like(actions) * 0.2)
            if ui_event.action == "exit" and ui_event.value > 0:
                # action_list = torch.stack(action_list)
                # dof_list = torch.stack(dof_list)
                # print("action_list:", action_list.shape)
                # print("dof_list:", dof_list.shape)
                # np.save("action_list.npy", action_list.cpu().numpy())
                # np.save("dof_list.npy", dof_list.cpu().numpy())

                print("exit")
                exit(0)
            if ui_event.action == "agent_full_reset" and ui_event.value > 0:
                print("agent_full_reset")
                agent_model.reset()
                estimator_model.reset()
            if ui_event.action == "full_reset" and ui_event.value > 0:
                print("full_reset")
                agent_model.reset()
                estimator_model.reset()
                print(env._get_terrain_curriculum_move([robot_index]))
                obs, _ = env.reset()
                env.commands[:, 2]=0
            if ui_event.action == "resample_commands" and ui_event.value > 0:
                print("resample_commands")
                env._resample_commands(torch.arange(env.num_envs, device= env.device))
            if ui_event.action == "stop" and ui_event.value > 0:
                env.commands[:, :3] = 0
                goal_command_mode=0
                # env.command_ranges["lin_vel_x"] = [0, 0]
                # env.command_ranges["lin_vel_y"] = [0, 0]
                # env.command_ranges["ang_vel_yaw"] = [0, 0]
            if ui_event.action == "forward" and ui_event.value > 0:
                env.commands[:, 0] = env_cfg.commands.ranges.lin_vel_x[1]
                goal_command_mode=1
                # env.command_ranges["lin_vel_x"] = [env_cfg.commands.ranges.lin_vel_x[1], env_cfg.commands.ranges.lin_vel_x[1]]
            if ui_event.action == "backward" and ui_event.value > 0:
                env.commands[:, 0] = env_cfg.commands.ranges.lin_vel_x[0]
                goal_command_mode=4
                # env.command_ranges["lin_vel_x"] = [env_cfg.commands.ranges.lin_vel_x[0], env_cfg.commands.ranges.lin_vel_x[0]]
            if ui_event.action == "leftward" and ui_event.value > 0:
                env.commands[:, 1] = env_cfg.commands.ranges.lin_vel_y[1]
                # env.command_ranges["lin_vel_y"] = [env_cfg.commands.ranges.lin_vel_y[1], env_cfg.commands.ranges.lin_vel_y[1]]
            if ui_event.action == "rightward" and ui_event.value > 0:
                env.commands[:, 1] = env_cfg.commands.ranges.lin_vel_y[0]
                # env.command_ranges["lin_vel_y"] = [env_cfg.commands.ranges.lin_vel_y[0], env_cfg.commands.ranges.lin_vel_y[0]]
            if ui_event.action == "leftturn" and ui_event.value > 0:
                env.commands[:, 2] = env_cfg.commands.ranges.ang_vel_yaw[1]
                goal_command_mode=2
                # env.command_ranges["ang_vel_yaw"] = [env_cfg.commands.ranges.ang_vel_yaw[1], env_cfg.commands.ranges.ang_vel_yaw[1]]
            if ui_event.action == "rightturn" and ui_event.value > 0:
                env.commands[:, 2] = env_cfg.commands.ranges.ang_vel_yaw[0]
                goal_command_mode=3
                # env.command_ranges["ang_vel_yaw"] = [env_cfg.commands.ranges.ang_vel_yaw[0], env_cfg.commands.ranges.ang_vel_yaw[0]]
            if ui_event.action == "leftdrag" and ui_event.value > 0:
                env.root_states[:, 7:10] += quat_rotate(env.base_quat, torch.tensor([[0., 0.5, 0.]], device= env.device))
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "rightdrag" and ui_event.value > 0:
                env.root_states[:, 7:10] += quat_rotate(env.base_quat, torch.tensor([[0., -0.5, 0.]], device= env.device))
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "mark" and ui_event.value > 0:
                mark = 0.05
            if ui_event.action == "env_left" and ui_event.value > 0:
                if env.terrain_types+1<env_cfg.terrain.num_cols:
                    env.terrain_types = env.terrain_types+1
                env.env_origins[:]=env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            if ui_event.action == "env_right" and ui_event.value > 0:
                if env.terrain_types>0:
                    env.terrain_types = env.terrain_types-1
                env.env_origins[:]=env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            if ui_event.action == "env_up" and ui_event.value > 0:
                if env.terrain_levels+1<env.max_terrain_level:
                    env.terrain_levels = env.terrain_levels+1
                env.env_origins[:]=env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            if ui_event.action == "env_down" and ui_event.value > 0:
                if env.terrain_levels>0:
                    env.terrain_levels = env.terrain_levels-1
                env.env_origins[:]=env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
            
        # if (env.contact_forces[robot_index, env.feet_indices, 2] > 200).any():
        #     print("contact_forces:", env.contact_forces[robot_index, env.feet_indices, 2])
        if (abs(env.substep_torques[robot_index]) > 35.).any():
            exceed_idxs = torch.where(abs(env.substep_torques[robot_index]) > 35.)
            print("substep_torques:", exceed_idxs[1], env.substep_torques[robot_index][exceed_idxs[0], exceed_idxs[1]])
        if i < stop_state_log:
            if torch.is_tensor(env.cfg.control.action_scale):
                action_scale = env.cfg.control.action_scale.detach().cpu().numpy()[joint_index]
            else:
                action_scale = env.cfg.control.action_scale
            base_roll = get_euler_xyz(env.base_quat)[0][robot_index].item()
            base_pitch = get_euler_xyz(env.base_quat)[1][robot_index].item()
            if base_pitch > torch.pi: base_pitch -= torch.pi * 2
            def reward_removed_term(term):
                return {"reward_removed_" + term: rews[robot_index].item() - (getattr(env, ("_reward_" + term))() * env.reward_scales[term])[robot_index].item()}
            log_states_dicts = {
                    # 'dof_pos_target': env.actions_scaled_torque_clipped[robot_index, joint_index].item(),
                    'dof_pos_target': env.actions[robot_index, joint_index].item() * action_scale,
                    'dof_pos': (env.dof_pos - env.default_dof_pos)[robot_index, joint_index].item(),
                    'dof_vel': env.substep_dof_vel[robot_index, 0, joint_index].max().item(),
                    'dof_torque': torch.mean(env.substep_torques[robot_index, :, joint_index]).item(),
                    # 'mark': obs_component[robot_index, 0].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_pitch': base_pitch,
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'max_torques': torch.abs(env.substep_torques).max().item(),
                    'max_torque_motor': torch.argmax(torch.max(torch.abs(env.substep_torques[robot_index]), dim= -2)[0], dim= -1).item() % 3, # between hip, thigh, calf
                    'max_torque_leg': int(torch.argmax(torch.max(torch.abs(env.substep_torques[robot_index]), dim= -2)[0], dim= -1).item() / 4), # between hip, thigh, calf
                    "action": actions[robot_index, joint_index].item(),
                    "reward": rews[robot_index].item(),
                    "power": torch.max(torch.sum(env.substep_torques * env.substep_dof_vel, dim= -1), dim= -1)[0][robot_index].item(),
                }
            if not "distill" in args.task:
                log_states_dicts.update(reward_removed_term("exceed_torque_limits_l1norm"))
                # log_states_dicts.update(reward_removed_term("legs_energy_substeps"))
                # log_states_dicts.update(reward_removed_term("ang_vel_xy"))
                # log_states_dicts.update(reward_removed_term("lin_vel_z"))
                # log_states_dicts.update(reward_removed_term("orientation"))
            logger.log_states(log_states_dicts)
        elif i==stop_state_log:
            # logger.plot_states()
            # import matplotlib.pyplot as plt
            # plt.plot(logger.state_log["dof_torque"], label="robot{}joint{}torque".format(robot_index, joint_index))
            # plt.legend()
            # plt.show()
            env._get_terrain_curriculum_move(torch.tensor([0], device= env.device))
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        if dones.any():
            agent_model.reset(dones)
            ppo_runner.alg.estimator.reset(dones)
            # if env.time_out_buf[dones].any():
            #     print("env dones, because has timeout")
            # else:
            #     print("env dones, because has fallen")
            # print(infos)
        # time.sleep(0.5)
        if i % 100 == 0:
            print("frame_rate:" , 100/(time.time_ns() - start_time) * 1e9, 
                  "command_x:", env.commands[robot_index, 0],
                  "root_state_x:", env.root_states[robot_index, 0],
            )
            start_time = time.time_ns()

        if env_cfg.env.goal_command:
            env._draw_goal_vis()

        # print(env.transition_buf)
        # print(ppo_runner.alg.dataset[0])
        # time.sleep(0.1)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    CAMERA_FOLLOW = MOVE_CAMERA
    args = get_args([
        dict(name= "--slow", type= float, default= 0., help= "slow down the simulation by sleep secs (float) every frame"),
        dict(name= "--zero_act_until", type= int, default= 0., help= "zero action until this step"),
        dict(name= "--sample", action= "store_true", default= False, help= "sample actions from policy"),
        dict(name= "--plot_time", type= int, default= 100, help= "plot states after this time"),
        dict(name= "--no_throw", action= "store_true", default= False),
    ])
    play(args)