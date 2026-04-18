import torch
import torch.nn as nn
from collections import namedtuple
import einops

from rsl_rl.modules.helpers import extract, apply_conditioning
import rsl_rl.utils.diffusion_utils as utils
torch.set_printoptions(threshold=float('inf'))

class PerTransitionRewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        """Legacy transition-level reward model."""
        super().__init__()
        input_dim = state_dim * 2 + action_dim
        dims = [input_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, s_i, s_ip1, a_i):
        """Compute transition-level reward."""
        x = torch.cat([s_i, s_ip1, a_i], dim=-1)
        r = self.network(x)
        return r.squeeze(-1)

class ValueGuide(nn.Module):
    """Guide wrapper that exposes value gradients."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        """Forward value estimate."""
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        """Return value and gradient w.r.t. input trajectory."""
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:
    """Guided trajectory sampler on top of diffusion model."""

    def __init__(self, guide, diffusion_model, **sample_kwargs):
        """Initialize guided sampler."""
        base_scale = float(sample_kwargs.pop("scale"))
        self._base_scale = base_scale
        
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.action_dim = diffusion_model.action_dim
        self.num_history = diffusion_model.num_history
        self.execute_num = diffusion_model.execute_num
        self.sample_kwargs = sample_kwargs
        self.scale = None
    
    @torch.no_grad()
    def resample_scale(self, env_mask, new_val = None):
        if self.scale is None or not env_mask.any():
            return
        if new_val is None:
            self.scale[env_mask] = self._base_scale
        else:
            self.scale[env_mask] = new_val

    def __call__(self, conditions, batch_size=1, num_candidate=1, verbose=True, candidate_reward_model=None, env_idx = None):
        """Sample trajectories and select candidate actions per environment."""
        
        if env_idx is None:
            scale_batch = self.scale
        else:
            scale_batch = self.scale[env_idx]
        scale_batch = scale_batch.view(-1, 1, 1)
        if num_candidate > 1:
            scale_batch = scale_batch.repeat_interleave(num_candidate, dim=0)

        for key in conditions:
            cond = conditions[key]
            cond = cond.unsqueeze(1).repeat(1, num_candidate, *([1] * (cond.dim() - 1)))
            conditions[key] = cond.view(-1, *cond.shape[2:])

        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, scale=scale_batch, **self.sample_kwargs)
        trajectories = samples.trajectories

        num_env = int(trajectories.shape[0] / num_candidate)
        horizon = trajectories.shape[1]
        transition_dim = trajectories.shape[2]
        trajectories = trajectories.view(num_env, num_candidate, horizon, transition_dim)

        if num_candidate == 1 or candidate_reward_model is None:
            best_indices = torch.zeros(num_env, dtype=torch.long, device=trajectories.device)
            candidate_scores = None
        else:
            batch_size_env, num_candidates, horizon_size, transition_size = trajectories.shape
            traj_bc = trajectories.reshape(batch_size_env * num_candidates, horizon_size, transition_size)

            t_zeros = torch.zeros(batch_size_env * num_candidates, dtype=torch.long, device=traj_bc.device)

            with torch.no_grad():
                scores = candidate_reward_model(traj_bc, None, t_zeros)

            if scores.dim() > 1:
                scores = scores.view(scores.size(0), -1).mean(dim=1)

            candidate_scores = scores.view(batch_size_env, num_candidates)
            best_indices = candidate_scores.argmax(dim=1)

        best_trajectories = trajectories[torch.arange(num_env), best_indices, :, :]

        actions = best_trajectories[:, self.num_history - 1 : self.num_history - 1 + self.execute_num, :self.action_dim]
        observations = best_trajectories[:, :, self.action_dim:]

        if (candidate_reward_model is not None) and (num_candidate > 1):
            selected_scores = candidate_scores[torch.arange(num_env), best_indices]
            trajectories_namedtuple = Trajectories(actions, observations, selected_scores)
        else:
            trajectories_namedtuple = Trajectories(actions, observations, None)
            
        return actions, trajectories_namedtuple

    @property
    def device(self):
        """Device of diffusion model parameters."""
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        """Convert and tile conditions for batched sampling."""
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(einops.repeat, conditions, 'd -> repeat d', repeat=batch_size)
        return conditions