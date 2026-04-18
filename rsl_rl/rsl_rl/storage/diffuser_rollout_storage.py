from collections import namedtuple
import torch
import numpy as np
from rsl_rl.utils import split_and_pad_trajectories
from rsl_rl.utils.buffer import buffer_from_example

class OnPolicyRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None

        def clear(self):
            self.__init__()

    MiniBatch = namedtuple("MiniBatch", [
        "obs",
        "actions",
        "values",
        "rewards",
        "avg_reward", 
        "weight",
    ])

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, actions_shape, device='cpu'):
        self.device = device
        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        Compute bootstrapped returns and advantages.
        Note: These are not used in the diffusion mini-batch.
        """
        advantage = 0
        self.returns = torch.zeros_like(self.rewards)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def diffusion_mini_batch_generator(
        self,
        mini_batch_size,
        min_traj_len,
        segment_length=20,
        num_epochs=8,
        segment_number=4096,
        temperature=None,
        filter_ratio=0.0,
    ):
        """
        Generate mini-batches for diffusion training, carrying along per-segment
        trajectory-average rewards and precomputed weights.

        filter_ratio: float in [0,1), fraction of trajectories with lowest
                    average reward to exclude.
        """
        padded_obs, traj_masks     = split_and_pad_trajectories(self.observations, self.dones)
        padded_actions, _          = split_and_pad_trajectories(self.actions, self.dones)
        padded_values, _           = split_and_pad_trajectories(self.values, self.dones)
        padded_rewards, _          = split_and_pad_trajectories(self.rewards, self.dones)

        traj_lengths = traj_masks.sum(dim=0)
        valid_traj_idx_tensor = (traj_lengths >= min_traj_len).nonzero(as_tuple=False).squeeze(1)
        valid_traj_idx = [int(x.item()) for x in valid_traj_idx_tensor]

        traj_avg_rewards = {}
        for idx in valid_traj_idx:
            L = int(traj_lengths[idx].item())
            total_r = padded_rewards[:L, idx].sum().item()
            traj_avg_rewards[idx] = total_r / L

        if 0.0 < filter_ratio < 1.0:
            avg_array = np.array(list(traj_avg_rewards.values()), dtype=np.float32)
            cutoff = np.quantile(avg_array, filter_ratio)
            filtered_idxs = [idx for idx, avg in traj_avg_rewards.items() if avg > cutoff]
            valid_traj_idx = filtered_idxs
            print(f"Filtered out {len(traj_avg_rewards) - len(filtered_idxs)} trajs below avg_reward cutoff {cutoff:.3f}")

        pool = []
        for idx in valid_traj_idx:
            L = int(traj_lengths[idx].item())
            if L < segment_length:
                continue
            for start_pos in range(0, L - segment_length + 1):
                pool.append((idx, start_pos))

        if len(pool) == 0:
            print("Warning: No trajectories meet the length requirement after filtering. No mini-batches generated.")
            return

        if len(pool) < segment_number:
            sampled_indices = np.arange(len(pool))
        else:
            sampled_indices = np.random.choice(len(pool), size=segment_number, replace=False)

        segments = []
        for i in sampled_indices:
            idx, start_pos = pool[i]
            seg = {
                "obs":        padded_obs[start_pos:start_pos+segment_length, idx].clone(),
                "actions":    padded_actions[start_pos:start_pos+segment_length, idx].clone(),
                "values":     padded_values[start_pos:start_pos+segment_length, idx].clone(),
                "rewards":    padded_rewards[start_pos:start_pos+segment_length, idx].clone(),
                "avg_reward": traj_avg_rewards[idx],
            }
            segments.append(seg)

        if temperature is not None:
            avg_array = np.array([s["avg_reward"] for s in segments], dtype=np.float32)
            r_max = avg_array.max()
            weights = np.exp((avg_array - r_max) / temperature)
            weights = weights / weights.sum() * len(weights)
        else:
            weights = np.ones(len(segments), dtype=np.float32)

        for seg, w in zip(segments, weights):
            seg["weight"] = float(w)

        for epoch in range(num_epochs):
            N = len(segments)
            permuted_idx = np.random.permutation(N)
            for start in range(0, N, mini_batch_size):
                batch_idx = permuted_idx[start:start + mini_batch_size]

                def stack_field(field):
                    return torch.stack([segments[i][field] for i in batch_idx], dim=1)

                avg_reward_batch = torch.tensor(
                    [segments[i]["avg_reward"] for i in batch_idx],
                    dtype=torch.float32
                )
                weight_batch = torch.tensor(
                    [segments[i]["weight"] for i in batch_idx],
                    dtype=torch.float32
                )

                yield OnPolicyRolloutStorage.MiniBatch(
                    obs=stack_field("obs"),
                    actions=stack_field("actions"),
                    values=stack_field("values"),
                    rewards=stack_field("rewards"),
                    avg_reward=avg_reward_batch,
                    weight=weight_batch,
                )

class OffPolicyRolloutStorage:
    """Infinite replay buffer + short-term cache for one rollout."""

    class Transition:
        """Single-timestep container."""
        def __init__(self):
            self.observations = None
            self.actions      = None
            self.rewards      = None
            self.dones        = None
            self.values       = None
            self.comp_rewards = None
            
        def clear(self):
            self.__init__()

    MiniBatch = namedtuple(
        "MiniBatch",
        ["obs", "actions", "values", "rewards", "comp_rewards", "avg_reward", "weight"]
    )

    def __init__(self,
                 num_envs: int,
                 num_steps_per_env: int,
                 obs_shape,
                 action_shape,
                 segment_length: int,
                 min_traj_len: int,
                 reward_names: list,
                 device: str = "cpu"):

        self.device = device
        self.max_steps = num_steps_per_env
        self.segment_length = segment_length
        self.min_traj_len = min_traj_len

        self.obs_buf    = torch.zeros(num_steps_per_env, num_envs, *obs_shape,    device=device)
        self.action_buf = torch.zeros(num_steps_per_env, num_envs, *action_shape, device=device)
        self.reward_buf = torch.zeros(num_steps_per_env, num_envs, 1,             device=device)
        self.done_buf   = torch.zeros(num_steps_per_env, num_envs, 1,
                                      dtype=torch.uint8, device=device)
        self.value_buf  = torch.zeros(num_steps_per_env, num_envs, 1,             device=device)
        
        self.reward_names = list(reward_names)
        self.name2idx = {n: i for i, n in enumerate(self.reward_names)}
        n_comp = len(self.reward_names)
        self.comp_reward_buf = torch.zeros(num_steps_per_env, num_envs, n_comp, device=device)
        self.step = 0

        self.replay_buffer = []

    def add_transitions(self, transition: Transition):
        if self.step >= self.max_steps:
            raise AssertionError("Rollout buffer overflow")
        self.obs_buf[self.step].copy_(transition.observations)
        self.action_buf[self.step].copy_(transition.actions)
        self.reward_buf[self.step].copy_(transition.rewards.view(-1, 1))
        self.done_buf[self.step].copy_(transition.dones.view(-1, 1))
        if transition.comp_rewards is not None:
            self.comp_reward_buf[self.step].copy_(transition.comp_rewards)
        self.step += 1

    def finish_rollout_and_update_replay(self):
        """
        Split cached transitions into full trajectories and append
        *all* of them (except those shorter than min_traj_len) to the replay buffer.
        *No reward‑based filtering here.*
        """
        padded_obs, traj_mask = split_and_pad_trajectories(self.obs_buf, self.done_buf)
        padded_actions, _     = split_and_pad_trajectories(self.action_buf, self.done_buf)
        padded_values, _      = split_and_pad_trajectories(self.value_buf, self.done_buf)
        padded_rewards, _     = split_and_pad_trajectories(self.reward_buf, self.done_buf)
        padded_comp_rew, _ = split_and_pad_trajectories(self.comp_reward_buf, self.done_buf)

        traj_lengths = traj_mask.sum(dim=0)
        env_ids      = range(traj_mask.shape[1])

        for i in env_ids:
            L = int(traj_lengths[i])
            if L < self.min_traj_len:
                continue
            self.replay_buffer.append(dict(
                obs     = padded_obs[:L, i].cpu(),
                actions = padded_actions[:L, i].cpu(),
                values  = padded_values[:L, i].cpu(),
                rewards = padded_rewards[:L, i].cpu(),
                dones   = torch.ones(L, 1, dtype=torch.uint8),
                comp_rewards = padded_comp_rew[:L, i].cpu(),
            ))
        print("replay buffer now have length:", len(self.replay_buffer))
        self._clear_cache()

    def diffusion_mini_batch_generator(self,
                                       mini_batch_size: int,
                                       traj_num: int,
                                       num_epochs: int,
                                       filter_ratio: float = 0.0,
                                       temperature: float = None):
        """
        1. Optionally drop the bottom `filter_ratio` fraction of trajectories
           w.r.t. average reward (done every call).
        2. Sample `traj_num` trajectories.
        3. Build a segment pool once; for each epoch, shuffle & yield mini‑batches.
        """
        if not self.replay_buffer:
            raise RuntimeError("Replay buffer is empty — collect rollouts first.")

        avg_rewards = np.array(
            [traj["rewards"].mean().item() for traj in self.replay_buffer],
            dtype=np.float32
        )

        signs = np.array(
            [np.sign(traj["obs"][0, 11].item()) for traj in self.replay_buffer],
            dtype=np.float32
        )

        pos_mask  = signs > 0
        neg_mask  = signs < 0
        zero_mask = signs == 0

        def _filter_by_reward(mask, name):
            """Return indices kept after reward filter within one sign class."""
            idx = np.where(mask)[0]
            if idx.size == 0:
                print(f"[Warn] no traj in class {name}.")
                return np.empty(0, dtype=int)
            if 0 < filter_ratio < 1:
                cutoff = np.quantile(avg_rewards[idx], filter_ratio)
                idx = idx[avg_rewards[idx] > cutoff]
            print(f"class {name}: kept {idx.size} traj")
            return idx

        pos_idx  = _filter_by_reward(pos_mask,  "pos")
        neg_idx  = _filter_by_reward(neg_mask,  "neg")
        zero_idx = _filter_by_reward(zero_mask, "zero")

        candidate_indices = np.concatenate([pos_idx, neg_idx, zero_idx])
        if candidate_indices.size == 0:
            raise RuntimeError("All trajectories filtered out. Consider lowering filter_ratio.")
        
        traj_num = min(traj_num, len(candidate_indices))
        sampled_idx = np.random.choice(candidate_indices, size=traj_num, replace=False)
        sampled_trajs = [self.replay_buffer[i] for i in sampled_idx]
        sampled_avg_r = avg_rewards[sampled_idx]
        sampled_signs = signs[sampled_idx].astype(np.int8)

        pool = []
        for t_idx, traj in enumerate(sampled_trajs):
            L = traj["obs"].shape[0]
            if L < self.segment_length:
                continue
            for s in range(0, L - self.segment_length + 1):
                pool.append((t_idx, s))
        if not pool:
            print("Warning: No segment available from sampled trajectories.")
            return

        for _ in range(num_epochs):
            np.random.shuffle(pool)
            for start in range(0, len(pool), mini_batch_size):
                batch_pairs = pool[start:start + mini_batch_size]

                def _stack(field):
                    return torch.stack(
                        [sampled_trajs[t][field][s:s + self.segment_length] for t, s in batch_pairs],
                        dim=1
                    ).to(self.device)

                obs_b     = _stack("obs")
                actions_b = _stack("actions")
                values_b  = _stack("values")
                rewards_b = _stack("rewards")
                comp_rew_b = _stack("comp_rewards")

                avg_b = torch.tensor(
                    [sampled_avg_r[t] for t, _ in batch_pairs],
                    dtype=torch.float32, device=self.device
                )

                avg_b_np = avg_b.cpu().numpy()

                if temperature is not None:
                    labels = sampled_signs[[t for t, _ in batch_pairs]]

                    w = np.empty(len(batch_pairs), dtype=np.float32)
                    temp = max(1e-8, float(temperature))

                    for cls in (-1, 0, 1):
                        idxs = np.where(labels == cls)[0]
                        if idxs.size == 0:
                            continue
                        class_vals = avg_b_np[idxs]
                        s = np.exp((class_vals - class_vals.max()) / temp)
                        s = s / s.sum() * idxs.size
                        w[idxs] = s
                else:
                    w = np.ones(len(batch_pairs), dtype=np.float32)

                weight = torch.tensor(w, dtype=torch.float32, device=self.device)

                yield OffPolicyRolloutStorage.MiniBatch(
                    obs       = obs_b,
                    actions   = actions_b,
                    values    = values_b,
                    rewards   = rewards_b,
                    comp_rewards=comp_rew_b,
                    avg_reward= avg_b,
                    weight    = weight,
                )

    def clear_low_reward_trajectories(self,
                                    clear_ratio: float,
                                    min_remaining: int = 1):
        """
        Remove the bottom `clear_ratio` fraction of trajectories w.r.t. average
        reward, but keep at least `min_remaining` trajectories in the buffer.

        Parameters
        ----------
        clear_ratio : float
            Fraction (0, 1) of worst trajectories to drop.
        min_remaining : int
            Minimal number of trajectories that must remain after pruning.
        """
        if clear_ratio <= 0.0 or clear_ratio >= 1.0 or not self.replay_buffer:
            return

        orig_size = len(self.replay_buffer)

        avg_rewards = np.array(
            [traj["rewards"].mean().item() for traj in self.replay_buffer],
            dtype=np.float32
        )
        cutoff = np.quantile(avg_rewards, clear_ratio)

        kept_trajs = [
            traj for traj, r in zip(self.replay_buffer, avg_rewards) if r > cutoff
        ]

        if len(kept_trajs) < min_remaining:
            sorted_pairs = sorted(
                zip(self.replay_buffer, avg_rewards),
                key=lambda x: x[1],
                reverse=True
            )
            kept_trajs = [traj for traj, _ in sorted_pairs[:min_remaining]]

        self.replay_buffer = kept_trajs

        print(f"[ReplayBuffer] Pruned {orig_size - len(self.replay_buffer)} / {orig_size} "
            f"trajectories; {len(self.replay_buffer)} remain.")

    def _clear_cache(self):
        """Reset the short-term rollout cache."""
        self.step = 0
        
