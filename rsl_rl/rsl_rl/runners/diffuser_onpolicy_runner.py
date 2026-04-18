import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import statistics
from collections import deque
import glob
import re

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import rsl_rl.modules as modules
from rsl_rl.env import VecEnv
from rsl_rl.modules.diffusion import GaussianDiffusion, ValueDiffusion
from rsl_rl.modules.temporal import TemporalUnet, ValueFunction
from rsl_rl.storage.diffuser_rollout_storage import OnPolicyRolloutStorage
from rsl_rl.algorithms.diffuser import GuidedPolicy, ValueGuide

class DiffuserOnPolicyRunner:
    def __init__(self, env: VecEnv, cfg, log_dir=None, device='cpu'):
        """Initialize on-policy diffusion runner."""
        self.cfg = cfg
        configured_num_candidate = int(self.cfg["policy"].get("num_candidate", 1))
        if configured_num_candidate != 1:
            print(f"[Release] Forcing policy.num_candidate from {configured_num_candidate} to 1.")
            self.cfg["policy"]["num_candidate"] = 1

        self.horizon = self.cfg["runner"]["horizon"]
        self.num_steps_per_env = self.cfg["runner"]["num_steps_per_env"]
        self.save_interval = self.cfg["runner"]["save_interval"]
        self.log_interval = self.cfg["runner"]["log_interval"]

        self.device = device
        self.env = env

        self.observation_dim = self.env.num_obs
        self.action_dim = self.env.num_actions
        self.max_grad_norm = self.cfg["algorithm"].get("max_grad_norm", 1.0)

        self.seed_steps = self.cfg["runner"]["seed_steps"]
        self.warmup_noise = self.cfg["runner"]["warmup_noise"]
        self.diff_obs_scale = torch.tensor(
            self.cfg["policy"].get("diff_obs_scale", [1.0] * self.observation_dim),
            device=self.device, dtype=torch.float32
        )
        self.diff_action_scale = torch.tensor(
            self.cfg["policy"].get("diff_action_scale", [1.0] * self.action_dim),
            device=self.device, dtype=torch.float32
        )

        self.diffuser_model = TemporalUnet(
            horizon=self.cfg["policy"]["horizon"],
            transition_dim=self.cfg["policy"]["transition_dim"],
            cond_dim=self.cfg["policy"]["cond_dim"],
            dim=self.cfg["policy"].get("dim", 32),
            dim_mults=self.cfg["policy"].get("dim_mults", (1, 2, 4, 8)),
            attention=self.cfg["policy"].get("attention", True),
        ).to(self.device)

        self.diffuser = GaussianDiffusion(
            self.diffuser_model,
            horizon=self.horizon,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_timesteps=self.cfg["runner"]["n_timesteps"],
            loss_type=self.cfg["runner"]["loss_type"],
            clip_denoised=self.cfg["runner"]["clip_denoised"],
            predict_epsilon=self.cfg["runner"]["predict_epsilon"],
            action_weight=self.cfg["runner"]["action_weight"],
            loss_discount=self.cfg["runner"]["loss_discount"],
            loss_weights=self.cfg["runner"]["loss_weights"],
            num_history=self.cfg["policy"]["num_history"],
            execute_num=self.cfg["policy"]["execute_num"],
            variance_clip=self.cfg["policy"]["variance_clip"],
        )
        
        self.reward_model = ValueDiffusion(
            model=ValueFunction(
                horizon=self.horizon,
                transition_dim=self.observation_dim + self.action_dim,
                cond_dim=self.observation_dim,
                dim=self.cfg["reward_model"]["full"]["dim"],
                dim_mults=self.cfg["reward_model"]["full"]["dim_mults"],
                out_dim=1,
            ),
            horizon=self.horizon,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_timesteps=self.cfg["runner"]["n_timesteps"],
            loss_type=self.cfg["runner"]["value_loss_type"],
            clip_denoised=self.cfg["runner"]["clip_denoised"],
            predict_epsilon=self.cfg["runner"]["predict_epsilon"],
            action_weight=self.cfg["runner"]["action_weight"],
            loss_discount=self.cfg["runner"]["loss_discount"],
            loss_weights=self.cfg["runner"]["loss_weights"],
        ).to(self.device)

        self.guide = ValueGuide(self.reward_model)
        self.guided_policy = GuidedPolicy(
            guide=self.guide,
            diffusion_model=self.diffuser,
            **self.cfg["guided_policy"]["sample_kwargs"]
        )
        self.guided_policy.scale = torch.full(  
                                            (env.num_envs, ),
                                            self.cfg["guided_policy"]["sample_kwargs"]["scale"],
                                            dtype = torch.float32,
                                            device = self.device)
        
        self.optimizer = torch.optim.Adam(
            self.diffuser.parameters(),
            lr=self.cfg["algorithm"]["learning_rate"]
        )
        self.reward_optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=self.cfg["algorithm"]["learning_rate"]
        )
        self.max_gradnorm = self.cfg["algorithm"].get("max_grad_norm", 1.0)

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10) if log_dir is not None else None
        print("Logging directory:", log_dir)

        self.tot_timesteps = 0
        self.tot_time = 0

        self.storage = OnPolicyRolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            obs_shape=[self.env.num_obs],
            actions_shape=[self.env.num_actions],
            device=self.device
        )
        self.env.reset()

        self.current_learning_iteration = 0
        
        self.planned_actions = None
        self.current_plan_indices = torch.zeros(self.env.num_envs, dtype=torch.int64, device=self.device)
        
        self.learn = self.learn_diffusion
        
        self.session_iterations = 0
        self.seed_steps = cfg["runner"]["seed_steps"]    

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
                action_noise_scale = self.cfg["policy"].get("action_noise_scale", None)
                if action_noise_scale is not None:
                    noise = torch.randn_like(planned_actions_subset) * action_noise_scale
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

        trans = OnPolicyRolloutStorage.Transition()
        trans.observations = obs
        trans.actions = action_batch
        trans.rewards = rewards
        trans.dones = dones
        self.storage.add_transitions(trans)

        return obs_next.to(self.device), rewards.to(self.device), dones.to(self.device), infos

    def learn_diffusion(self, num_learning_iterations):
        """Main training loop for diffusion and reward models."""
        tot_iter = self.current_learning_iteration + num_learning_iterations

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        while self.current_learning_iteration < tot_iter:
            iter_start_time = time.time()

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

            if self.session_iterations < self.seed_steps:
                learn_start = time.time()
                mean_losses = {"diffusion_loss": 0.0, "reward_loss": 0.0}
                learn_time = time.time() - learn_start
            else:
                mean_losses = {"diffusion_loss": 0., "reward_loss": 0.}
                num_batches = 0
                learn_start = time.time()
                for minibatch in self.storage.diffusion_mini_batch_generator(
                    mini_batch_size=self.cfg["runner"]["mini_batch_size"],
                    min_traj_len=self.cfg["runner"]["min_traj_len"],
                    segment_length=self.cfg["runner"]["segment_length"],
                    num_epochs=self.cfg["runner"]["diffusion_training_epochs"],
                    segment_number=self.cfg["runner"]["segment_number"],
                    temperature=self.cfg["algorithm"]["temperature"],
                    filter_ratio=self.cfg["algorithm"]["filter_ratio"],
                ):
                    scaled_obs = minibatch.obs * self.diff_obs_scale
                    scaled_actions = minibatch.actions * self.diff_action_scale
                    batch_transitions = torch.cat([scaled_actions, scaled_obs], dim=-1).to(self.device).transpose(0, 1)
                    batch_cond = minibatch.obs[0].to(self.device) * self.diff_obs_scale
                    batch_cond = {0: batch_cond}

                    target_reward = minibatch.rewards.mean(dim=0).to(self.device)
                    reward_weights = minibatch.weight.to(self.device)

                    loss_diff, _ = self.diffuser.loss(
                        batch_transitions,
                        batch_cond,
                        reward_weights=reward_weights
                    )
                    t = torch.randint(
                        0,
                        self.reward_model.n_timesteps,
                        (minibatch.obs.shape[1],),
                        device=self.device
                    ).long()
                    loss_rew, _ = self.reward_model.p_losses(
                        batch_transitions,
                        batch_cond,
                        target_reward,
                        t,
                    )

                    losses = dict(
                        diffusion_loss=loss_diff,
                        reward_loss=loss_rew,
                    )
                    total_loss = sum(getattr(self, f"{k}_coef", 1.) * v for k, v in losses.items())
                    for k in mean_losses.keys():
                        mean_losses[k] += losses[k].detach()
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.diffuser.parameters(), self.max_gradnorm)
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_gradnorm)
                    self.optimizer.step()
                    num_batches += 1
                
                if num_batches > 0:
                    for k in mean_losses.keys():
                        mean_losses[k] /= num_batches
                
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
            self.storage.clear()
            self.current_learning_iteration += 1
            self.session_iterations += 1 
            if self.log_dir is not None \
                and self.current_learning_iteration % self.save_interval == 0 \
                and self.session_iterations > self.seed_steps:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        
    def log(self, locs, width=80, pad=35):
        """
        Log training statistics, including losses, performance metrics, reward stats,
        and episode info to TensorBoard and console.
        Expected keys in locs:
          - collection_time, learn_time, tot_iter, losses
          - ep_infos: list of episode info dictionaries.
          - rframebuffer: per-timestep rewards.
          - rewbuffer: per-episode total rewards.
          - lenbuffer: per-episode lengths.
          - start_iter: iteration at which logging started.
        """
        iteration_time = locs['collection_time'] + locs['learn_time']
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += iteration_time

        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time) if iteration_time > 0 else 0

        # Log losses.
        for k, v in locs["losses"].items():
            self.writer.add_scalar("Loss/" + k, v.item() if isinstance(v, torch.Tensor) else v, self.current_learning_iteration)

        # Log performance metrics.
        try:
            self.writer.add_scalar("Perf/gpu_allocated", torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
            self.writer.add_scalar("Perf/gpu_occupied", torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        except Exception:
            pass
        self.writer.add_scalar("Perf/total_fps", fps, self.current_learning_iteration)
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], self.current_learning_iteration)
        self.writer.add_scalar("Perf/learn_time", locs["learn_time"], self.current_learning_iteration)
        self.writer.add_scalar("Perf/total_timesteps", self.tot_timesteps, self.current_learning_iteration)
        self.writer.add_scalar("Perf/total_time", self.tot_time, self.current_learning_iteration)

        # Log reward statistics.
        if len(locs.get("rframebuffer", [])) > 0:
            mean_reward_timestep = statistics.mean(locs["rframebuffer"])
            self.writer.add_scalar("Train/mean_reward_each_timestep", mean_reward_timestep, self.current_learning_iteration)
        if len(locs.get("rewbuffer", [])) > 0:
            mean_reward = statistics.mean(locs["rewbuffer"])
            self.writer.add_scalar("Train/mean_reward", mean_reward, self.current_learning_iteration)
            self.writer.add_scalar("Train/mean_reward_time", mean_reward, self.tot_time)
        if len(locs.get("lenbuffer", [])) > 0:
            mean_episode_length = statistics.mean(locs["lenbuffer"])
            self.writer.add_scalar("Train/mean_episode_length", mean_episode_length, self.current_learning_iteration)
            self.writer.add_scalar("Train/mean_episode_length_time", mean_episode_length, self.tot_time)

        # Log episode info.
        ep_string = ""
        if "ep_infos" in locs and locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                info_tensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    value = ep_info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor([value], device=self.device)
                    elif value.dim() == 0:
                        value = value.unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, value.to(self.device)))
                mean_value = torch.mean(info_tensor)
                self.writer.add_scalar("Episode/" + key, mean_value, self.current_learning_iteration)
                ep_string += f"{('Mean episode ' + key + ':'):>{pad}} {mean_value:.4f}\n"

        header = f" Learning iteration {self.current_learning_iteration}/{locs.get('tot_iter', 'N/A')} "
        log_string = f"\n{'#' * width}\n"
        log_string += f"{header.center(width, ' ')}\n\n"
        log_string += f"{'Computation:':>{pad}} {fps} steps/s (collection: {locs['collection_time']:.3f}s, learning: {locs['learn_time']:.3f}s)\n"

        for loss_key in ["diffusion_loss", "reward_loss"]:
            if loss_key in locs["losses"]:
                log_string += f"{loss_key.replace('_', ' ').capitalize() + ':':>{pad}} {locs['losses'][loss_key]:.4f}\n"
        
        if len(locs['rewbuffer']) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""

        log_string += ep_string
        log_string += f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
        log_string += f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
        log_string += f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        try:
            remaining = locs["tot_iter"] - self.current_learning_iteration
            eta = self.tot_time / (self.current_learning_iteration + 1) * remaining
            log_string += f"{'ETA:':>{pad}} {eta:.1f}s\n"
        except Exception:
            pass
        log_string += f"{'-' * width}\n"
        print(log_string)

    def save(self, path, infos=None):
        """Save diffuser, reward model, and optimizer state."""
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