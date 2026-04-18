import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import statistics
from collections import deque
import importlib
import glob
import re
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

import rsl_rl.modules as modules
from rsl_rl.env import VecEnv
from rsl_rl.modules.diffusion import GaussianDiffusion, ValueDiffusion
from rsl_rl.modules.temporal import TemporalUnet, ValueFunction
from rsl_rl.storage.diffuser_rollout_storage import OnPolicyRolloutStorage, OffPolicyRolloutStorage
from rsl_rl.runners.diffuser_onpolicy_runner import DiffuserOnPolicyRunner
from rsl_rl.algorithms.diffuser import GuidedPolicy, ValueGuide, PerTransitionRewardModel




class DiffuserOffPolicyRunner(DiffuserOnPolicyRunner):

    def __init__(self, env, cfg, log_dir=None, device='cpu'):

        super().__init__(env, cfg, log_dir, device)
        configured_num_candidate = int(self.cfg["policy"].get("num_candidate", 1))
        if configured_num_candidate != 1:
            print(f"[Release] Forcing policy.num_candidate from {configured_num_candidate} to 1.")
            self.cfg["policy"]["num_candidate"] = 1
        self.storage = OffPolicyRolloutStorage(               
            num_envs = env.num_envs,
            num_steps_per_env = self.num_steps_per_env,
            obs_shape = [env.num_obs],
            action_shape = [env.num_actions],
            segment_length = cfg["runner"]["segment_length"],
            min_traj_len = cfg["runner"]["min_traj_len"],
            reward_names=env.reward_names,
            device = device,
        )
        self.learn = self.learn_offpolicy_diffusion   
        self.session_iterations = 0
        self.seed_steps = self.cfg["runner"]["seed_steps"]
        self.action_noise_cfg = self.cfg["policy"].get("action_noise", None)
        if self.action_noise_cfg and self.action_noise_cfg.get("schedule_type") == "custom":
            mod, fn = self.action_noise_cfg["custom_schedule_func"].rsplit(".", 1)
            self._custom_noise_fn = getattr(importlib.import_module(mod), fn)
        else:
            self._custom_noise_fn = None
            
        self.update_interval = self.cfg["runner"]["update_interval"]
            
    def rollout_step(self, obs):
        """Run one rollout step and store transitions."""
        num_envs = obs.shape[0]

        if (self.session_iterations < self.seed_steps and self.warmup_noise is not None):
            action_batch = (
                torch.randn(num_envs, self.action_dim, device=self.device)
                * self.warmup_noise
            )
        else:
            conditions = {0: obs * self.diff_obs_scale}

            if self.planned_actions is None:
                replan_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
            else:
                replan_mask = self.current_plan_indices >= self.cfg["policy"]["execute_num"]

            if replan_mask.any():
                indices = replan_mask.nonzero(as_tuple=False).squeeze(-1)
                conditions_subset = {0: conditions[0][indices]}
                planned_actions_subset, _ = self.guided_policy(
                    conditions_subset,
                    batch_size=len(indices),
                    num_candidate=1,
                    verbose=False,
                    candidate_reward_model=None,
                    env_idx=indices,
                )
                planned_actions_subset = torch.tensor(planned_actions_subset, device=self.device)
                noise_scale = self._get_action_noise_scale()
                if noise_scale > 0:
                    noise = torch.randn_like(planned_actions_subset) * noise_scale
                    planned_actions_subset = planned_actions_subset + noise
                diff_action_scale_tensor = torch.tensor(
                    self.cfg["policy"]["diff_action_scale"],
                    device=self.device,
                    dtype=planned_actions_subset.dtype
                )
                planned_actions_subset = planned_actions_subset / diff_action_scale_tensor

                if self.planned_actions is None:
                    self.planned_actions = torch.zeros(num_envs, self.cfg["policy"]["execute_num"], self.action_dim, device=self.device)
                self.planned_actions[indices] = planned_actions_subset
                self.current_plan_indices[indices] = 0

            action_batch = self.planned_actions[torch.arange(num_envs), self.current_plan_indices, :]
            self.current_plan_indices += 1

        obs_next, _, rewards, dones, infos = self.env.step(action_batch)

        done_mask = (dones > 0)
        if done_mask.any():
            self.current_plan_indices[done_mask] = self.cfg["policy"]["execute_num"]
            if self.cfg["guided_policy"]["resample_guidance_scale"]:
                new_scale = torch.rand(done_mask.sum(), device=self.device) * self.guided_policy._base_scale
                self.guided_policy.resample_scale(done_mask, new_scale)

        trans = OffPolicyRolloutStorage.Transition()
        trans.observations = obs
        trans.actions = action_batch
        trans.rewards = rewards
        trans.comp_rewards = None
        trans.dones = dones
        self.storage.add_transitions(trans)

        return obs_next.to(self.device), rewards.to(self.device), dones.to(self.device), infos
    
    def _get_action_noise_scale(self) -> float:
        """Return scalar noise scale according to current session_iterations."""
        if not self.action_noise_cfg: 
            return self.cfg["policy"].get("action_noise_scale", 0.0)

        base = self.action_noise_cfg.get("base_scale", 0.0)
        stype = self.action_noise_cfg.get("schedule_type", "constant")
        params = self.action_noise_cfg.get("schedule_params", {})

        if stype == "constant":
            return base

        it = self.session_iterations

        if stype == "linear":
            t = (it - params.get("start_iter", 0)) / max(
                params.get("end_iter", 1) - params.get("start_iter", 0), 1
            )
            t = max(0.0, min(1.0, t))
            end_scale = params.get("end_scale", 0.0)
            return base + t * (end_scale - base)

        if stype == "exponential":
            decay = params.get("decay_rate", 0.999)
            res = base * (decay ** it)
            clipped_res = max(res, params["floor"])
            return clipped_res
            
        if stype == "custom" and self._custom_noise_fn:
            return self._custom_noise_fn(it, base, params)

        return base

    def learn_offpolicy_diffusion(self, num_learning_iterations):
        """Main off-policy training loop for diffusion and reward models."""
        tot_iter = self.current_learning_iteration + num_learning_iterations

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        while self.current_learning_iteration < tot_iter:

            obs = self.env.get_observations().to(self.device)
            rollout_start = time.time()
            for _ in range(self.num_steps_per_env):
                obs, rewards, dones, infos = self.rollout_step(obs)
                if 'episode' in infos:
                    ep_infos.append(infos['episode'])
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                if new_ids.numel() > 0:
                    rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            collection_time = time.time() - rollout_start
            learn_start = time.time()
            self.storage.finish_rollout_and_update_replay()
            
            clear_cycle = self.cfg["runner"]["clear_cycle"]
            clear_ratio = self.cfg["runner"]["clear_ratio"]
            min_traj_after_clear = self.cfg["runner"]["min_traj_after_clear"]
            if clear_cycle and clear_cycle > 0 \
                    and (self.session_iterations + 1) % clear_cycle == 0 \
                    and clear_ratio > 0.0:
                self.storage.clear_low_reward_trajectories(
                    clear_ratio=clear_ratio,
                    min_remaining=min_traj_after_clear
                )
            
            skip_training_phase = (self.session_iterations < self.seed_steps) \
                      or (self.session_iterations % self.update_interval != 0)
            
            if skip_training_phase:
                mean_losses = {"diffusion_loss": 0.0, "reward_model_loss": 0.0}

                reward_batches = 0
                reward_loss_total = 0.0
                
                for mb in self.storage.diffusion_mini_batch_generator(
                        mini_batch_size=self.cfg["reward_model"]["full"]["batch_size"],
                        traj_num=self.cfg["reward_model"]["full"]["traj_num"],
                        num_epochs=self.cfg["reward_model"]["full"]["num_training_epochs"]):
            
                    scaled_obs     = mb.obs     * self.diff_obs_scale
                    scaled_actions = mb.actions * self.diff_action_scale
                    batch_trans   = torch.cat([scaled_actions, scaled_obs], dim=-1)\
                                         .to(self.device).transpose(0,1)
                    batch_cond     = {0: scaled_obs[0]}

                    t = torch.randint(
                        0,
                        self.cfg["runner"]["n_timesteps"],
                        (mb.obs.shape[1],),
                        device=self.device
                    ).long()

                    target_reward = mb.rewards.mean(dim=0).to(self.device)
                    loss_r, _ = self.reward_model.p_losses(batch_trans, batch_cond, target_reward, t)

                    self.reward_optimizer.zero_grad()
                    loss_r.backward()
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_gradnorm)
                    self.reward_optimizer.step()

                    reward_loss_total += loss_r.item()
                    
                    reward_batches += 1

                avg_reward_loss = reward_loss_total / reward_batches if reward_batches > 0 else 0.0
                mean_losses["reward_model_loss"] = avg_reward_loss

            else:
                mean_losses = {"diffusion_loss": 0., "reward_model_loss": 0.}
                num_batches = 0
                for mb in self.storage.diffusion_mini_batch_generator(
                        mini_batch_size=self.cfg["runner"]["mini_batch_size"],
                        traj_num       =self.cfg["runner"]["traj_num"],
                        num_epochs     =self.cfg["runner"]["diffusion_training_epochs"],
                        filter_ratio   =self.cfg["algorithm"]["filter_ratio"],
                        temperature    =self.cfg["algorithm"]["temperature"]):
                    scaled_obs     = mb.obs     * self.diff_obs_scale
                    scaled_actions = mb.actions * self.diff_action_scale
                    batch_trans   = torch.cat([scaled_actions, scaled_obs], dim=-1)\
                                         .to(self.device).transpose(0,1)
                    batch_cond     = {0: scaled_obs[0]}
                    

                    loss_diff, _ = self.diffuser.loss(batch_trans, batch_cond,
                                                      reward_weights=mb.weight.to(self.device))

                    mean_losses["diffusion_loss"] += loss_diff.detach()
                    self.optimizer.zero_grad()
                    loss_diff.backward()
                    torch.nn.utils.clip_grad_norm_(self.diffuser.parameters(), self.max_gradnorm)
                    self.optimizer.step()
                    num_batches += 1

                if num_batches:
                    mean_losses["diffusion_loss"] /= num_batches

                reward_batches     = 0
                reward_loss_total  = 0.0
                
                for mb in self.storage.diffusion_mini_batch_generator(
                        mini_batch_size=self.cfg["reward_model"]["full"]["batch_size"],
                        traj_num       =self.cfg["reward_model"]["full"]["traj_num"],
                        num_epochs     =self.cfg["reward_model"]["full"]["num_training_epochs"]):

                    scaled_obs     = mb.obs     * self.diff_obs_scale
                    scaled_actions = mb.actions * self.diff_action_scale
                    batch_trans   = torch.cat([scaled_actions, scaled_obs], dim=-1)\
                                         .to(self.device).transpose(0,1)
                    batch_cond     = {0: scaled_obs[0]}
                    t = torch.randint(0, self.cfg["runner"]["n_timesteps"],
                                      (mb.obs.shape[1],), device=self.device).long()
                    target_reward = mb.rewards.mean(dim=0).to(self.device)
                    loss_r, _ = self.reward_model.p_losses(
                        batch_trans, batch_cond, target_reward, t
                    )

                    self.reward_optimizer.zero_grad()
                    loss_r.backward()
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_gradnorm)
                    self.reward_optimizer.step()

                    reward_loss_total += loss_r.item()
                    reward_batches += 1

                avg_reward_loss = reward_loss_total/reward_batches if reward_batches else 0.0

                mean_losses["reward_model_loss"] = avg_reward_loss
                
                if num_batches > 0:
                    mean_losses["diffusion_loss"] /= num_batches
                
            learn_time = time.time() - learn_start

            log_dict = {
                "collection_time": collection_time,
                "learn_time": learn_time,
                "tot_iter": tot_iter,
                "losses": mean_losses,
                "ep_infos": ep_infos,
                "rframebuffer": list(rframebuffer),
                "rewbuffer": list(rewbuffer),
                "lenbuffer": list(lenbuffer),
                "start_iter": self.current_learning_iteration,
            }
            if self.writer is not None:
                self.log(log_dict)
            self.current_learning_iteration += 1
            self.session_iterations += 1 
            if self.log_dir is not None \
                and self.current_learning_iteration % self.save_interval == 0 \
                and self.session_iterations > self.seed_steps:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            
    def save(self, path, infos=None):
        """
        Save the state of the diffuser/reward models and optimizers.
        """
        run_state_dict = {
            'diffuser_state_dict': self.diffuser.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        torch.save(run_state_dict, path)
        print(f"Saved model at iteration {self.current_learning_iteration} to {path}")

    def load(self, path, load_optimizer=True):
        """Load diffuser and reward model states."""
        loaded_dict = torch.load(path, map_location=self.device)
        self.diffuser.load_state_dict(loaded_dict['diffuser_state_dict'], strict=False)
        print("variance", self.diffuser.posterior_variance)
        rm_cfg = self.cfg.get("reward_model", {})

        if rm_cfg.get("load_external", False):
            logs_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "rewards")

            reward_subdir = rm_cfg["reward_path"]
            base_dir = (reward_subdir if os.path.isabs(reward_subdir)
                        else os.path.join(logs_root, reward_subdir))

            rew_ckpt = rm_cfg.get("rew_ckpt", -1)
            if rew_ckpt is not None and rew_ckpt >= 0:
                candidate_path = os.path.join(base_dir, f"model_{rew_ckpt}.pt")
            else:
                files = glob.glob(os.path.join(base_dir, "model_*.pt"))
                if not files:
                    raise FileNotFoundError(f"No reward checkpoints in {base_dir}")
                def _idx(f):
                    m = re.search(r"model_(\d+)\.pt$", os.path.basename(f))
                    return int(m.group(1)) if m else -1
                candidate_path = max(files, key=_idx)

            print(f"Loading external reward model: {candidate_path}")
            ext_state = torch.load(candidate_path, map_location=self.device)
            state = (ext_state.get("reward_model_state_dict")
                    if isinstance(ext_state, dict) and "reward_model_state_dict" in ext_state
                    else ext_state)
            self.reward_model.load_state_dict(state, strict=True)
        else:
            self.reward_model.load_state_dict(
                loaded_dict['reward_model_state_dict'], strict=False
            )

        self.guide = ValueGuide(self.reward_model)
        self.guided_policy = GuidedPolicy(
            guide=self.guide,
            diffusion_model=self.diffuser,
            **self.cfg["guided_policy"]["sample_kwargs"]
        )
        self.guided_policy.scale = torch.full(  
                                            (self.env.num_envs,),
                                            self.cfg["guided_policy"]["sample_kwargs"]["scale"],
                                            dtype = torch.float32,
                                            device = self.device)
        self.current_learning_iteration = loaded_dict['iter']
        print(f"Loaded model from {path} at iteration {self.current_learning_iteration}")
        return loaded_dict.get('infos', None)

