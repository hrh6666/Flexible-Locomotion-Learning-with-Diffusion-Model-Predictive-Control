from collections import namedtuple
import numpy as np
import torch
from torch import nn
from rsl_rl.modules.constraints import ConstraintContext
from rsl_rl.utils.telemetry import DiffusionTelemetry

from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, constraints=None,**kwargs,
):
    """Perform n-step guided DDPM update for one reverse timestep."""
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)
    y = None

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        grad[t < t_stopgrad] = 0
        
        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)
        if constraints is not None:
            ctx = ConstraintContext(
                transition_dim=x.shape[-1],
                action_dim=model.action_dim,
            )
            x = constraints(x, ctx)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    noise = torch.randn_like(x)
    noise[t == 0] = 0
    
    clipped_model_std = model_std.clamp(min=model.variance_clip)
    x_next = model_mean + clipped_model_std * noise

    return x_next, y


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=30,
        loss_type='l1',
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        num_history = 1,
        execute_num = 1,
        variance_clip = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.num_history = num_history
        self.execute_num = execute_num
        self.variance_clip = variance_clip
        print("num_history:", num_history)
        print("execute_num:", execute_num)

        self.device = next(model.parameters()).device

        betas = cosine_beta_schedule(n_timesteps).to(self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.predict_epsilon = predict_epsilon
        
        self._cache_enabled = False
        self._cache_m = 0
        self._cached_x = None
        self._cached_i = None
        
        self._telemetry_enabled = False
        self._telemetry = None

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_variance = posterior_variance.clamp_min(self.variance_clip)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ddim_A = torch.sqrt((1 - alphas_cumprod_prev) / (1 - alphas_cumprod))
        ddim_B = torch.sqrt(1 - alphas_cumprod / alphas_cumprod_prev)
        ddim_B[alphas_cumprod_prev == 0] = 0
        self.register_buffer('ddim_A', ddim_A)
        self.register_buffer('ddim_B', ddim_B)

        self.register_buffer('sqrt_alphas_cumprod_prev', torch.sqrt(alphas_cumprod_prev))

        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        Set loss coefficients for the trajectory.
        
        Parameters:
            action_weight: Coefficient for the first action loss.
            discount: Multiplicative discount factor for the t-th timestep (discount**t).
            weights_dict: Dictionary {i: c} to scale the observation loss for dimension i.
        
        Returns:
            loss_weights: A tensor of loss weights.
        """
        self.action_weight = action_weight
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32, device=self.device)
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float, device=self.device)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights[self.num_history-1, :self.action_dim] = action_weight
        return loss_weights

    def predict_start_from_noise(self, x_t, t, noise):
        """Recover x0 estimate from current sample and predicted noise."""
        if self.predict_epsilon:
            return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """Compute q(x_{t-1} | x_t, x0) parameters."""
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        """Compute p(x_{t-1} | x_t) mean/variance terms."""
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def ddim_sample_fn(
        diffusion, x, cond, t, eta=0.0,
        guide=None, scale=0.001, t_stopgrad=0,
        n_guide_steps=1, scale_grad_by_std=True,
        constraints=None,
    ):
        """Perform one guided DDIM update."""
        y = None
        if guide is not None:
            model_log_variance = extract(diffusion.posterior_log_variance_clipped, t, x.shape)
            model_var = torch.exp(model_log_variance)
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    y, grad = guide.gradients(x, cond, t)
                if scale_grad_by_std:
                    grad = model_var * grad
                grad[t < t_stopgrad] = 0
                x = x + scale * grad
                x = apply_conditioning(x, cond, diffusion.action_dim)
                if constraints is not None:
                    ctx = ConstraintContext(
                        transition_dim=x.shape[-1],
                        action_dim=diffusion.action_dim,
                    )
                    x = constraints(x, ctx)

        eps = diffusion.model(x, cond, t)
        x0_pred = diffusion.predict_start_from_noise(x, t, eps)
        alpha_bar_t = extract(diffusion.alphas_cumprod, t, x.shape)
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_prev = extract(diffusion.alphas_cumprod, t_prev, x.shape)
        ddim_A_t = extract(diffusion.ddim_A, t, x.shape)
        ddim_B_t = extract(diffusion.ddim_B, t, x.shape)
        sigma_t = eta * ddim_A_t * ddim_B_t
        eps_pred = (x - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1 - alpha_bar_t)
        noise = torch.randn_like(x) if eta > 0.0 else 0.0
        x_prev = (
            extract(diffusion.sqrt_alphas_cumprod_prev, t, x.shape) * x0_pred +
            torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred +
            sigma_t * noise
        )
        x_prev = apply_conditioning(x_prev, cond, diffusion.action_dim)
        values = torch.zeros(x.shape[0], device=x.device)
        return x_prev, y
    
    @torch.no_grad()
    def ddim_sample_loop(self, shape, cond, eta=0.0, seq=None, verbose=True, return_chain=False, **sample_fn_kwargs):
        """Generate trajectories with DDIM sampling."""
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)
        chain = [x] if return_chain else None

        if seq is None:
            seq = list(range(self.n_timesteps))
        else:
            seq = sorted(seq)
        for i in reversed(seq[1:]):
            t = make_timesteps(batch_size, i, device)
            x, values = self.ddim_sample_fn(x, cond, t, eta=eta, **sample_fn_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)
            if return_chain:
                chain.append(x)
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)
    

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=n_step_guided_p_sample, **sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]

        use_cached = (self._cache_enabled and (self._cached_x is not None) and (self._cached_i is not None))
        if use_cached:
            x = self._cached_x.clone().to(device)
            x = apply_conditioning(x, cond, self.action_dim)
            start_i = int(self._cached_i)
            print("Using cached sample at i = {} for early-denoising.".format(start_i))       
        else:
            x = torch.randn(shape, device=device)
            x = apply_conditioning(x, cond, self.action_dim)
            start_i = self.n_timesteps - 1

        chain = [x] if return_chain else None

        for i in reversed(range(0, start_i + 1)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)
            
            if self._telemetry_enabled and (self._telemetry is not None):
                self._telemetry.maybe_log(x, cond, t)

            if self._cache_enabled and self._cache_m > 0 and not use_cached:
                boundary_i = self.n_timesteps - 1 - int(self._cache_m)
                if i == boundary_i and i > 0:
                    self._cached_x = x.detach()
                    self._cached_i = i - 1

            if return_chain:
                chain.append(x)
        
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)
    
    def set_early_cache(self, enable: bool = False, m_steps: int = 0):
        """Enable or disable early-denoising cache."""
        self._cache_enabled = bool(enable)
        self._cache_m = int(m_steps)
        if not self._cache_enabled or self._cache_m <= 0:
            self._cached_x = None
            self._cached_i = None

    def invalidate_cache(self):
        """Clear any cached intermediate sample."""
        self._cached_x = None
        self._cached_i = None

    def conditional_sample(self, cond, horizon=None, sample_type="ddpm", skip_type=None, timesteps=None, **sample_kwargs):
        """Dispatch conditional sampling with DDPM or DDIM backend."""
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        if sample_type == "ddpm":
            return self.p_sample_loop(shape, cond, **sample_kwargs)
        elif sample_type in ["ddim"]:
            seq = None
            if skip_type == "uniform":
                skip = self.n_timesteps // timesteps
                seq = list(range(0, self.n_timesteps, skip))
            elif skip_type == "quad":
                seq = np.linspace(0, np.sqrt(self.n_timesteps * 0.8), timesteps) ** 2
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError("Unsupported skip_type: {}".format(skip_type))
            return self.ddim_sample_loop(shape, cond, seq=seq, **sample_kwargs)
        else:
            raise NotImplementedError("Unsupported sample_type: {}".format(sample_type))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                  extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        return sample

    def p_losses(self, x_start, cond, t, reward_weights=None):
        """Compute diffusion loss for one noisy timestep."""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, reward_weights=reward_weights)
        else:
            loss, info = self.loss_fn(x_recon, x_start, reward_weights=reward_weights)
        return loss, info

    def loss(self, x, *args, reward_weights=None):
        """Sample random timestep and compute training loss."""
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t, reward_weights=reward_weights)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
    
    def enable_telemetry(self, out_dir: str, **kwargs):
        self._telemetry_enabled = True
        self._telemetry = DiffusionTelemetry(out_dir, **kwargs)

    def disable_telemetry(self):
        self._telemetry_enabled = False
        if self._telemetry is not None:
            self._telemetry.flush()
        self._telemetry = None

class ValueDiffusion(GaussianDiffusion):
    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        pred = self.model(x_noisy, cond, t)
        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)