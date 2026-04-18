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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

import rsl_rl.algorithms as algorithms
import rsl_rl.modules as modules
from rsl_rl.env import VecEnv
import torch.nn.functional as F


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.device = device
        self.env = env
        
        actor_critic = modules.build_actor_critic(
            self.env,
            self.cfg["policy_class_name"],
            self.policy_cfg,
        ).to(self.device)
        
        estimator = modules.Estimator(input_dim=self.estimator_cfg["input_dim"], output_dim=actor_critic.estimator_output_dim, 
                                      rnn_type=self.estimator_cfg["rnn_type"],rnn_hidden_size=self.estimator_cfg["rnn_hidden_size"],
                                      latent_encoder_hidden_dims=self.estimator_cfg["latent_encoder_hidden_dims"]).to(self.device)
        
        if hasattr(actor_critic.actor, "num_props"):
            actor_critic.actor.num_props=3+self.estimator_cfg["input_dim"]
        else:
            actor_critic.num_props=3+self.estimator_cfg["input_dim"]

        discriminator = modules.Discriminator(state_size=self.discriminator_cfg["state_size"], hidden_sizes=self.discriminator_cfg["hidden_sizes"])
        
        alg_class = getattr(algorithms, self.cfg["algorithm_class_name"]) # PPO
        if self.cfg["algorithm_class_name"] == "PPO":
            self.alg: algorithms.PPO = alg_class(actor_critic, estimator, discriminator, self.estimator_cfg, self.discriminator_cfg, device=self.device, **self.alg_cfg)
        else:
            self.alg: algorithms.RPPO = alg_class(actor_critic, estimator, discriminator, self.estimator_cfg, self.discriminator_cfg, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        
        self.amp_reward_coeff = self.cfg.get("amp_reward_coeff", 0.)
        self.num_estimator_train_step = self.cfg.get("num_estimator_train_step", False)
        self.stage = self.cfg.get("stage", False)
        self.learn = self.learn_RL
        self.learn_and_record = self.learn_Record
        
        self.save_interval = self.cfg["save_interval"]
        self.estimator_decay = self.estimator_cfg["estimator_decay"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions],
                              [actor_critic.estimator_output_dim], [1], [self.discriminator_cfg["state_size"]*2])
        
        # terrain_types=self.env.terrain_types.clone().detach().to(self.device)
        # ### TODO: Change hard coded values
        # terrain_types[terrain_types==0]=0
        # terrain_types[terrain_types==1]=0
        # terrain_types[terrain_types==2]=0
        # terrain_types[terrain_types==3]=1
        # terrain_types[terrain_types==4]=1
        # terrain_types[terrain_types==5]=2
        # terrain_types[terrain_types==6]=2
        # terrain_types[terrain_types==7]=2
        # terrain_types[terrain_types==8]=3
        # terrain_types[terrain_types==9]=3
        # self.alg.storage.env_classify[:,:,0]=terrain_types
        
        terrain_types = self.env.terrain_types.clone().detach().to(self.device)

        terrain_cfg = self.env.cfg.terrain
        num_cols = terrain_cfg.num_cols
        if(hasattr(terrain_cfg, 'col_bar')):
            # Define the order and labels for terrain types
            # 0: col_bar, 1: col_baffle 2: col_pit, 3: col_stair, 4: col_slope, 5: col_pole
            terrain_labels_order = [0, 1, 2, 3, 4, 5]
            terrain_counts = [
                terrain_cfg.col_bar,
                terrain_cfg.col_baffle,
                terrain_cfg.col_pit,
                terrain_cfg.col_stair,
                terrain_cfg.col_slope,
                terrain_cfg.col_pole
            ]

            empty_ground_label = 6  # Label for empty ground

            # Create a mapping list based on terrain counts
            mapping = []
            for label, count in zip(terrain_labels_order, terrain_counts):
                mapping.extend([label] * count)

            remaining = num_cols - len(mapping)
            if remaining < 0:
                raise ValueError("Total terrain counts exceed num_rows in the configuration.")

            mapping.extend([empty_ground_label] * remaining)

            # Convert the mapping list to a tensor for efficient indexing
            # Shape: [num_cols]
            mapping_tensor = torch.tensor(mapping, device=terrain_types.device)

            # Ensure that terrain_types indices are within the valid range [0, num_cols - 1]
            if terrain_types.max() >= num_cols:
                raise ValueError(
                    f"terrain_types contains index {terrain_types.max()} which exceeds num_rows={num_cols - 1}."
                )
            terrain_labels_mapped = mapping_tensor[terrain_types]
            self.alg.storage.env_classify[:, :, 0] = terrain_labels_mapped
        
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)

        _, _ = self.env.reset()
    
    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.estimator.train()

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.estimator_iteration = 0
        self.gt_ratio = 1.
        while self.current_learning_iteration < tot_iter:
            start = time.time()
            self.alg.gt_ratio=self.gt_ratio
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()
            if self.estimator_cfg["train_together"] and self.estimator_iteration>self.estimator_cfg["decay_start_step"]:
                self.gt_ratio = self.gt_ratio * self.estimator_decay
            self.estimator_iteration = self.estimator_iteration + 1
            self.current_learning_iteration = self.current_learning_iteration + 1
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration))) 

    def learn_Record(self, num_learning_iterations, init_at_random_ep_len=False):
        # Initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  
        self.alg.estimator.train()

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.estimator_iteration = 0
        self.gt_ratio = 1.

        # Dynamically construct grad_logging_modules, including gating_network and each Expert module in the ModuleList
        grad_logging_modules = {
            'actor': self.alg.actor_critic.actor
        }

        # {module_name: {param_name: {'sum': tensor, 'count': int}}}
        accumulated_gradients = {}
        for m_name, module in grad_logging_modules.items():
            accumulated_gradients[m_name] = {}
            for p_name, p in module.named_parameters():
                if p.requires_grad:
                    accumulated_gradients[m_name][p_name] = {
                        'sum': torch.zeros_like(p.data, dtype=torch.float, device=self.device),
                        'count': 0
                    }
                else:
                    accumulated_gradients[m_name][p_name] = None

        while self.current_learning_iteration < tot_iter:
            start = time.time()
            self.alg.gt_ratio = self.gt_ratio

            # Rollout
            with torch.no_grad():  
                for i in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            self.alg.compute_returns(critic_obs)

            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start

            for m_name, module in grad_logging_modules.items():
                for p_name, p in module.named_parameters():
                    if not p.requires_grad:
                        continue  
                    if accumulated_gradients[m_name][p_name] is None:
                        continue  
                    if p.grad is not None:
                        accumulated_gradients[m_name][p_name]['sum'] += p.grad.detach().clone()
                    accumulated_gradients[m_name][p_name]['count'] += 1

            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            ep_infos.clear()
            if self.estimator_cfg["train_together"] and self.estimator_iteration > self.estimator_cfg["decay_start_step"]:
                self.gt_ratio = self.gt_ratio * self.estimator_decay
            self.estimator_iteration += 1
            self.current_learning_iteration += 1

            if self.current_learning_iteration % 50 == 0:
                avg_grad_checkpoint = {}
                for m_name, params_dict in accumulated_gradients.items():
                    avg_grad_checkpoint[m_name] = {}
                    for p_name, grad_info in params_dict.items():
                        if grad_info is not None:
                            if grad_info['count'] > 0:
                                avg_grad_checkpoint[m_name][p_name] = grad_info['sum'] / grad_info['count']
                            else:
                                avg_grad_checkpoint[m_name][p_name] = torch.zeros_like(grad_info['sum'])
                        else:
                            avg_grad_checkpoint[m_name][p_name] = None
                grad_checkpoint_file = os.path.join(self.log_dir, f'gradient_checkpoint_iter_{self.current_learning_iteration}.pt')
                torch.save(avg_grad_checkpoint, grad_checkpoint_file)

        avg_grad_dict = {}
        for m_name, params_dict in accumulated_gradients.items():
            avg_grad_dict[m_name] = {}
            for p_name, grad_info in params_dict.items():
                if grad_info is not None:
                    if grad_info['count'] > 0:
                        avg_grad_dict[m_name][p_name] = grad_info['sum'] / grad_info['count']
                    else:
                        avg_grad_dict[m_name][p_name] = torch.zeros_like(grad_info['sum'])
                else:
                    avg_grad_dict[m_name][p_name] = None

        avg_grad_file = os.path.join(self.log_dir, f'average_gradients_iter_{self.current_learning_iteration}.pt')
        torch.save(avg_grad_dict, avg_grad_file)
        # Save the final model
        # self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))
        
    def rollout_step(self, obs, critic_obs):
        actions = self.alg.act(obs, critic_obs)
        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
        transition = self.env.transition_buf
        self.alg.update_transition(transition)
        with torch.inference_mode():
            amp_score=self.alg.discriminator(transition)
        amp_rew=torch.clip(1-0.25*(amp_score-1)**2, min=0)*self.amp_reward_coeff
        amp_rew=amp_rew.squeeze(1)

        rewards=rewards+amp_rew
        infos['episode']['rew_amp']=amp_rew
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        self.alg.process_env_step(rewards, dones, infos)
        return obs, critic_obs, rewards, dones, infos

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, self.current_learning_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        for k, v in locs["losses"].items():
            self.writer.add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        for k, v in locs["stats"].items():
            self.writer.add_scalar("Train/" + k, v.item(), self.current_learning_iteration)
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.current_learning_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_allocated', torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_occupied', torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        if self.estimator_cfg["train_together"]:
            self.writer.add_scalar('Train/gt_ratio', self.gt_ratio, self.current_learning_iteration)
        self.writer.add_scalar('Train/mean_reward_each_timestep', statistics.mean(locs['rframebuffer']), self.current_learning_iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n""")
        print(log_string)
        
        
    def log_estimator(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, self.current_learning_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        for k, v in locs["losses"].items():
            self.writer.add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        # for k, v in locs["stats"].items():
        #     self.writer.add_scalar("Train/" + k, v.item(), self.current_learning_iteration)
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        # self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.current_learning_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_allocated', torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_occupied', torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        if self.estimator_cfg["train_together"]:
            self.writer.add_scalar('Train/gt_ratio', self.gt_ratio, self.current_learning_iteration)
        self.writer.add_scalar('Train/mean_reward_each_timestep', statistics.mean(locs['rframebuffer']), self.current_learning_iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Gate loss:':>{pad}} {locs["losses"]['gate_loss']:.4f}\n"""
                          f"""{'Latent loss:':>{pad}} {locs["losses"]['latent_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n""")
        print(log_string)    

    def save(self, path, infos=None):
        run_state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.alg.discriminator_optimizer.state_dict(),
            # 'estimator_optimizer_state_dict': self.alg.estimator_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if hasattr(self.alg, "lr_scheduler"):
            run_state_dict["lr_scheduler_state_dict"] = self.alg.lr_scheduler.state_dict()
        torch.save(run_state_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        print("Loaded State Dict Keys:")
        for key in loaded_dict['model_state_dict'].keys():
            print(key)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'],strict=False)
        if "estimator_state_dict" in loaded_dict:
            self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'], strict=False)
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # if load_optimizer and "optimizer_state_dict" in loaded_dict:
        #     self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        if "discriminator_optimizer_state_dict" in loaded_dict:
            self.alg.discriminator_optimizer.load_state_dict(loaded_dict['discriminator_optimizer_state_dict'])
            # if "estimator_optimizer_state_dict" in loaded_dict:
            #     self.alg.estimator_optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        if "lr_scheduler_state_dict" in loaded_dict:
            if not hasattr(self.alg, "lr_scheduler"):
                print("Warning: lr_scheduler_state_dict found in checkpoint but no lr_scheduler in algorithm. Ignoring.")
            else:
                self.alg.lr_scheduler.load_state_dict(loaded_dict["lr_scheduler_state_dict"])
        elif hasattr(self.alg, "lr_scheduler"):
            print("Warning: lr_scheduler_state_dict not found in checkpoint but lr_scheduler in algorithm. Ignoring.")
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']
    
    def load_moe(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        saved_state_dict = loaded_dict['model_state_dict']
        new_state_dict = {}

        # Get the number of experts in your current model
        num_experts = self.alg.actor_critic.expert_num

        # Copy and adjust actor parameters
        actor_keys = [k for k in saved_state_dict.keys() if k.startswith('actor.')]
        critic_keys = [k for k in saved_state_dict.keys() if k.startswith('critic.')]

        # Map 'actor.*' parameters to 'actor.{i}.actor.actor_backbone.*' for each expert
        for i in range(num_experts):
            for k in actor_keys:
                # Extract the parameter name after 'actor.'
                param_name = k[len('actor.'):]
                # Construct the new key
                new_key = f'actor.{i}.actor.actor_backbone.{param_name}'
                new_state_dict[new_key] = saved_state_dict[k]

        # Map 'critic.*' parameters to 'critic.{i}.critic.*' for each expert
        for i in range(num_experts):
            for k in critic_keys:
                param_name = k[len('critic.'):]
                new_key = f'critic.{i}.critic.{param_name}'
                new_state_dict[new_key] = saved_state_dict[k]

        # Copy the remaining parameters as they are
        for k in saved_state_dict.keys():
            if not k.startswith('actor.') and not k.startswith('critic.'):
                new_state_dict[k] = saved_state_dict[k]

        # Load the adjusted state_dict into your current model
        self.alg.actor_critic.load_state_dict(new_state_dict, strict=False)

        # Load other components if they exist
        if "estimator_state_dict" in loaded_dict:
            self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'], strict=False)
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)

        # Optionally load optimizer states
        if load_optimizer:
            # We do not load the optimizer of the model
            if "discriminator_optimizer_state_dict" in loaded_dict:
                self.alg.discriminator_optimizer.load_state_dict(loaded_dict['discriminator_optimizer_state_dict'])

        # Load learning rate scheduler state if it exists
        if "lr_scheduler_state_dict" in loaded_dict:
            if hasattr(self.alg, "lr_scheduler"):
                self.alg.lr_scheduler.load_state_dict(loaded_dict["lr_scheduler_state_dict"])
            else:
                print("Warning: lr_scheduler_state_dict found in checkpoint but no lr_scheduler in algorithm. Ignoring.")
        elif hasattr(self.alg, "lr_scheduler"):
            print("Warning: lr_scheduler_state_dict not found in checkpoint but lr_scheduler in algorithm. Ignoring.")

        # Update the current learning iteration
        self.current_learning_iteration = loaded_dict.get('iter', 0)

        return loaded_dict.get('infos', None)
    
    def load_actor(self, idx, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        debug=self.alg.actor_critic.actor[idx].load_state_dict(loaded_dict['model_state_dict'],strict=False)
        return loaded_dict['infos']
    
    def load_critic(self, idx, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.critic[idx].load_state_dict(loaded_dict['model_state_dict'],strict=False)
        # self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
