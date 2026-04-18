import torch
import torch.nn as nn
import einops

ACTION_DIM = 12
OBS_DIM    = 49
OBS_OFF    = ACTION_DIM  # global offset for all obs slices within the transition vector

def split_obs(x):
    """
    Split the *trajectory* vector into named OBS slices with a +12 shift for actions.
    x: [B, H, T] where the first 12 dims are the current action (not returned),
       followed by 49-D observation laid out as documented above.
    Returns:
        v_lin, v_ang, g_proj, cmd, q, qd, u_prev, mu
        (each is [B, H, *])
    """
    v_lin  = x[..., OBS_OFF + 0 : OBS_OFF + 3]
    v_ang  = x[..., OBS_OFF + 3 : OBS_OFF + 6]
    g_proj = x[..., OBS_OFF + 6 : OBS_OFF + 9]
    cmd    = x[..., OBS_OFF + 9 : OBS_OFF +12]
    q      = x[..., OBS_OFF +12 : OBS_OFF +24]
    qd     = x[..., OBS_OFF +24 : OBS_OFF +36]
    u_prev = x[..., OBS_OFF +36 : OBS_OFF +48]  # last actions from OBS
    mu     = x[..., OBS_OFF +48 : OBS_OFF +49]
    return v_lin, v_ang, g_proj, cmd, q, qd, u_prev, mu

# Unitree Go2 per-leg order in q/qd/u_prev:
# ['FL_hip_joint','FL_thigh_joint','FL_calf_joint',
#  'FR_hip_joint','FR_thigh_joint','FR_calf_joint',
#  'RL_hip_joint','RL_thigh_joint','RL_calf_joint',
#  'RR_hip_joint','RR_thigh_joint','RR_calf_joint']
LEG_SLICES = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}
JOINT_IN_LEG = {"hip_joint": 0, "thigh_joint": 1, "calf_joint": 2}
EPS = 1e-8  # small constant to avoid division by zero

def mean_over_time(x):
    """
    Reduce over horizon and any remaining feature dims to [B, 1].
    Works for shapes like [B,H,D], [B,H,1], or [B,H].
    """
    while x.dim() > 3:
        x = x.mean(dim=-1, keepdim=True)
    if x.dim() == 3:
        x = x.mean(dim=1, keepdim=True)
    elif x.dim() == 2:
        x = x.mean(dim=1, keepdim=True)
    return x

# --- tiny helper (minimal change) ---
def _normalize_legs(leg_or_legs):
    """Return a non-empty list of leg tags like ['FL','FR'].
    Accepts a single string ('FL') or a list/tuple of strings."""
    if isinstance(leg_or_legs, (list, tuple)):
        assert len(leg_or_legs) > 0, "legs list must be non-empty"
        return list(leg_or_legs)
    else:
        return [leg_or_legs]

class DofVelRew(nn.Module):
    """
    Reward function that penalises the L2 norm (mean-squared value) of
    DoF velocities in dimensions 36-48 of the transition vector.

    Input signature is kept identical to `ValueFunction.forward`:
        x    : [batch, horizon, transition_dim]
        cond : (unused, kept for API compatibility)
        time : (unused, kept for API compatibility)
        *args: (ignored, kept for API compatibility)

    Output:
        reward : [batch, 1]   # higher is better (negative penalty)
    """
    def __init__(self, w = 1.0):
        super().__init__()
        self.w = w

    def forward(self, x, cond, time, *args):
        # ensure shape [B, H, T] → [B, H, 12], T = dim_action + dim_observation = 12 + 49 = 61
        vel_slice = x[:, 1:, 36:48]
        mse_penalty = (vel_slice.abs()).mean(dim=(1, 2), keepdim=True)
        
        pos_slice = x[:, 1:, 24:36]
        vel = pos_slice[:, 1:, :] - pos_slice[:, :-1, :]
        mse_penalty += self.w * (vel.abs()).mean(dim=(1, 2), keepdim=True)
        
        reward = -mse_penalty
        return reward
    

class DofAccRew(nn.Module):
    def forward(self, x, cond, time, *args):
        vel = x[:, 1:, 36:48]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        mse_penalty = (acc ** 2).mean(dim=(1, 2), keepdim=True)
        return -mse_penalty

class DofPosRew(nn.Module):
    def forward(self, x, cond, time, *args):
        pos_slice = x[:, 1:, 24:36]
        mse_penalty = (pos_slice ** 2).mean(dim=(1, 2), keepdim=True)
        reward = -mse_penalty
        return reward

class DofRewBipOrientation(nn.Module):
    """
    Reward encouraging biped orientation using the projected gravity vector.

    Input signature matches ValueFunction.forward:
        x    : [batch, horizon, transition_dim]
        cond : (unused, kept for API compatibility)
        time : (unused, kept for API compatibility)
        *args: (ignored)

    Implementation:
        - Uses projected gravity from x[..., 6:9] for each timestep.
        - Computes cosine similarity with a target direction [-1, 0, 0].
        - Maps cosine in [-1, 1] to [0, 1] via (0.5 * cos + 0.5), then squares it
          to sharpen the peak and penalize large misalignment more strongly.
        - Averages the per-timestep rewards over the horizon to return [batch, 1].
    Output:
        reward : [batch, 1]   (higher is better)
    """

    def __init__(self, target_dir=None, eps: float = 1e-8):
        super().__init__()
        # Register the target direction as a buffer for proper device/dtype handling
        if target_dir is None:
            target_dir = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
        self.register_buffer("target_dir", target_dir)
        self.eps = eps

    def forward(self, x, cond=None, time=None, *args):
        # x: [B, H, T]; projected gravity slice -> [B, H, 3]
        proj_g = x[: ,1: , 18:21]

        # Ensure target direction broadcasts to [B, H, 3]
        # (the buffer is [3], unsqueeze for broadcast)
        target = self.target_dir.view(1, 1, 3).to(dtype=x.dtype, device=x.device)

        # Cosine similarity between projected gravity and target direction
        dot = (proj_g * target).sum(dim=-1)                    # [B, H]
        g_norm = torch.norm(proj_g, dim=-1)                    # [B, H]
        t_norm = torch.norm(target, dim=-1)                    # [1, 1]
        # Prevent division by zero; t_norm is constant (==1 if target is unit), but keep general
        cos_theta = dot / (g_norm * t_norm + self.eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)          # numerical safety

        # Map cos in [-1, 1] to [0, 1], then square to produce a sharper reward peak
        step_rew = (0.5 * cos_theta + 0.5) ** 2                # [B, H]

        # Average across horizon to get a single reward per batch element
        reward = step_rew.mean(dim=1, keepdim=True)            # [B, 1]
        return reward
    
class LegPosExcursionRew(nn.Module):
    """
    Penalize posture excursion around q_ref for one or MULTIPLE legs:
        r = -E_t[ mean_{legs} mean_{joints} | q_leg(t) - q_ref | ]  (L1 by default; set l2=True for L2).
    Args:
        leg or legs: "FL"/"FR"/"RL"/"RR" or a list of them.
        q_ref: [3] for per-leg joints (broadcasted to all selected legs) or [L,3] to specify per-leg refs.
    """
    def __init__(self, leg="FL", q_ref=None, l2=False):
        super().__init__()
        self.legs = _normalize_legs(leg)
        self.l2 = l2
        # keep a generic buffer; shape normalized in forward for device/dtype
        if q_ref is None:
            q_ref = torch.zeros(3)
        self.register_buffer('q_ref_buf', q_ref.clone().detach())

    def forward(self, x, cond, time, *args):
        x = x[:, 1:, :]
        *_, q, _, _ = split_obs(x)  # q: [B,H,12]

        # gather q for selected legs into [B,H,L,3]
        segs = [q[..., LEG_SLICES[L]] for L in self.legs]
        q_sel = torch.stack(segs, dim=-2)  # [B,H,L,3]

        # build q_ref with shape [1,1,L,3]
        if self.q_ref_buf.dim() == 1:  # [3]
            q_ref = self.q_ref_buf.view(1,1,1,3).to(q_sel)
            q_ref = q_ref.expand(-1,-1,len(self.legs),-1)
        else:  # [L,3]
            assert self.q_ref_buf.size(0) == len(self.legs), "q_ref shape must match #legs"
            q_ref = self.q_ref_buf.view(1,1,len(self.legs),3).to(q_sel)

        # L1 or squared L2 over joints, then average over legs
        if self.l2:
            err = (q_sel - q_ref).pow(2).mean(dim=-1, keepdim=True)  # [B,H,L,1]
        else:
            err = (q_sel - q_ref).abs().mean(dim=-1, keepdim=True)   # [B,H,L,1]

        # average over legs, then over time → [B,1]
        err = err.mean(dim=-2, keepdim=True)  # [B,H,1,1]
        return -mean_over_time(err.squeeze(-1))  # [B,1]

class LegVelUsageRew(nn.Module):
    """
    Penalize velocity magnitude on one or MULTIPLE legs:
        r = -E_t[ mean_{legs} ||qd_leg(t)||^2 ]  (or subset of joints).
    """
    def __init__(self, leg="FL", joints=None):
        super().__init__()
        self.legs = _normalize_legs(leg)
        self.joints = joints

    def forward(self, x, cond, time, *args):
        x = x[:, 1:, :]
        *_, qd, _, _ = split_obs(x)  # [B,H,12]

        segs = [qd[..., LEG_SLICES[L]] for L in self.legs]          # list of [B,H,3]
        qd_sel = torch.stack(segs, dim=-2)                           # [B,H,L,3]
        if self.joints is not None:
            idx = torch.as_tensor([JOINT_IN_LEG[j] for j in self.joints],
                                  dtype=torch.long, device=qd_sel.device)
            qd_sel = qd_sel.index_select(-1, idx)                    # [B,H,L,len(joints)]

        mse = (qd_sel ** 2).mean(dim=(-1, -2), keepdim=True)         # mean over joints & legs → [B,H,1]
        return -mean_over_time(mse)                                  # [B,1]

class LegAccUsageRew(nn.Module):
    """
    Penalize acceleration (velocity differences) on one or MULTIPLE legs:
        r = -E_t[ mean_{legs} ||Δqd_leg(t)||^2 ].
    """
    def __init__(self, leg="FL", joints=None):
        super().__init__()
        self.legs = _normalize_legs(leg)
        self.joints = joints

    def forward(self, x, cond, time, *args):
        x = x[:, 1:, :]
        *_, qd, _, _ = split_obs(x)
        segs = [qd[..., LEG_SLICES[L]] for L in self.legs]           # [B,H,3]
        qd_sel = torch.stack(segs, dim=-2)                            # [B,H,L,3]
        if self.joints is not None:
            idx = torch.as_tensor([JOINT_IN_LEG[j] for j in self.joints],
                                  dtype=torch.long, device=qd_sel.device)
            qd_sel = qd_sel.index_select(-1, idx)                     # [B,H,L,len(joints)]
        acc = qd_sel[:, 1:, ...] - qd_sel[:, :-1, ...]                # [B,H-1,L,*]
        mse = (acc ** 2).mean(dim=(-1, -2), keepdim=True)             # [B,H-1,1]
        return -mean_over_time(mse)

class LegVelShareRew(nn.Module):
    """
    Penalize the fractional velocity usage of SELECTED leg(s) vs. all legs:
        numerator  = sum_{selected legs} ||qd_leg(t)||   (sum of per-leg magnitudes)
        denominator= ||qd_all(t)||                       (magnitude over all 12 DoFs)
        r = -E_t[ (numerator / (denominator + EPS))^2 ].
    """
    def __init__(self, leg="FL"):
        super().__init__()
        self.legs = _normalize_legs(leg)

    def forward(self, x, cond, time, *args):
        x = x[:, 1:, :]
        *_, qd, _, _ = split_obs(x)                           # [B,H,12]

        # sum of magnitudes over the selected legs
        mags = []
        for L in self.legs:
            mags.append(qd[..., LEG_SLICES[L]].norm(dim=-1, keepdim=True))  # [B,H,1]
        num = torch.stack(mags, dim=-1).sum(dim=-1)                          # [B,H,1]

        denom = qd.norm(dim=-1, keepdim=True)                                 # [B,H,1]
        frac = (num / (denom + EPS)) ** 2
        return -mean_over_time(frac)                                          # [B,1]
    
class LegPosShareRew(nn.Module):
    """
    Penalize fractional posture excursion of SELECTED leg(s) vs. all legs:
        numerator  = sum_{selected legs} || q_leg - q_ref_leg ||
        denominator= || q_all - q_ref_all ||                 (refs default to 0)
        r = -E_t[ (numerator / (denominator + EPS))^2 ].
    q_ref: [3] (broadcast to selected legs) or [L,3].
    """
    def __init__(self, leg="FL", q_ref=None):
        super().__init__()
        self.legs = _normalize_legs(leg)
        if q_ref is None:
            q_ref = torch.zeros(3)
        self.register_buffer('q_ref_buf', q_ref.clone().detach())

    def forward(self, x, cond, time, *args):
        x = x[:, 1:, :]
        *_, q, _, _ = split_obs(x)  # [B,H,12]

        # Build per-leg references for selected legs into [1,1,L,3]
        if self.q_ref_buf.dim() == 1:  # [3]
            q_ref_sel = self.q_ref_buf.view(1,1,1,3).to(q)
            q_ref_sel = q_ref_sel.expand(-1,-1,len(self.legs),-1)
        else:  # [L,3]
            assert self.q_ref_buf.size(0) == len(self.legs), "q_ref shape must match #legs"
            q_ref_sel = self.q_ref_buf.view(1,1,len(self.legs),3).to(q)

        # numerator: sum of norms over selected legs
        q_sel = torch.stack([q[..., LEG_SLICES[L]] for L in self.legs], dim=-2)  # [B,H,L,3]
        num = (q_sel - q_ref_sel).norm(dim=-1, keepdim=True).sum(dim=-2)         # [B,H,1]

        # denominator: all DoFs vs all-legs refs (0 by default)
        q_ref_all = torch.zeros_like(q)
        # If user provided non-zero refs only for selected legs, fill them in:
        for i, L in enumerate(self.legs):
            if self.q_ref_buf.dim() == 1:
                q_ref_all[..., LEG_SLICES[L]] = self.q_ref_buf.view(1,1,3).to(q)
            else:
                q_ref_all[..., LEG_SLICES[L]] = self.q_ref_buf[i].view(1,1,3).to(q)

        denom = (q - q_ref_all).norm(dim=-1, keepdim=True)                       # [B,H,1]
        frac = (num / (denom + EPS)) ** 2
        return -mean_over_time(frac)                                             # [B,1]
    

# -------- Joint-level constraints (q / qd only) --------

class JointSoftRangeRew(nn.Module):
    """
    Soft box constraint for specified joints on selected legs:
        penalty = ReLU(q - hi)^2 + ReLU(lo - q)^2
        r = -E_t[ penalty ].

    Args:
        joints: tuple/list like ("hip_joint",) or ("hip_joint","thigh_joint","calf_joint")
        legs:   list like ["FL","FR","RL","RR"]; if None, apply to all legs.
        lo, hi:
            - scalar (backward compatible), e.g., lo=-0.3, hi=0.3
            - dict by joint name, e.g., lo={"hip_joint":-0.2, "thigh_joint":-0.5}
            - dict by (leg, joint), e.g., hi={("FL","hip_joint"):0.4, ("FR","hip_joint"):0.35}
            - nested dict by leg -> joint, e.g., lo={"FL":{"hip_joint":-0.25}}
        Resolution priority (most specific first):
            (leg, joint)  >  leg->{joint}  >  {joint}  >  scalar default
    """
    def __init__(self, joints=("hip_joint",), legs=None, lo=-0.3, hi=0.3):
        super().__init__()
        self.joints = joints
        self.legs = ["FL","FR","RL","RR"] if legs is None else legs
        # keep raw specs to allow flexible resolution in forward
        self.lo_spec = lo
        self.hi_spec = hi

    @staticmethod
    def _resolve_bound(spec, leg, joint, default):
        """Return a float bound for (leg, joint) given a spec (scalar or dicts)."""
        # scalar
        if isinstance(spec, (int, float)):
            return float(spec)
        # torch scalar
        if torch.is_tensor(spec) and spec.dim() == 0:
            return float(spec.item())

        # dict-based resolution
        if isinstance(spec, dict):
            # 1) (leg, joint) pair
            if (leg, joint) in spec:
                v = spec[(leg, joint)]
                return float(v if not torch.is_tensor(v) else v.item())
            # 2) nested: spec[leg][joint]
            if leg in spec and isinstance(spec[leg], dict) and (joint in spec[leg]):
                v = spec[leg][joint]
                return float(v if not torch.is_tensor(v) else v.item())
            # 3) by joint name
            if joint in spec:
                v = spec[joint]
                return float(v if not torch.is_tensor(v) else v.item())

        # fallback
        return float(default)

    def forward(self, x, cond, time, *args):
        x = x[:, 1:, :]
        *_, q, _, _ = split_obs(x)  # q: [B,H,12]

        # build channel indices and per-channel lo/hi arrays
        idxs, lo_list, hi_list = [], [], []
        for leg in self.legs:
            base = LEG_SLICES[leg].start
            for j in self.joints:
                idxs.append(base + JOINT_IN_LEG[j])
                lo_val = self._resolve_bound(self.lo_spec, leg, j, default=-0.3)
                hi_val = self._resolve_bound(self.hi_spec, leg, j, default= 0.3)
                # (optional) ensure lo <= hi to avoid negative infeasible band
                if lo_val > hi_val:
                    lo_val, hi_val = hi_val, lo_val
                lo_list.append(lo_val)
                hi_list.append(hi_val)

        idxs = torch.as_tensor(idxs, dtype=torch.long, device=q.device)
        sel  = q.index_select(-1, idxs)  # [B,H,C], C = len(self.legs) * len(self.joints)

        lo_t = torch.tensor(lo_list, dtype=q.dtype, device=q.device).view(1,1,-1)  # [1,1,C]
        hi_t = torch.tensor(hi_list, dtype=q.dtype, device=q.device).view(1,1,-1)  # [1,1,C]

        over  = torch.relu(sel - hi_t)
        under = torch.relu(lo_t - sel)
        pen = (over + under).mean(dim=-1, keepdim=True)  # [B,H,1]
        return -mean_over_time(pen)  # [B,1]

class JointVelLimitRew(nn.Module):
    """
    Soft velocity cap for specified joints/legs using a logistic margin:
        r = E_t[ sigmoid(k * (vcap - |qd|)) ] averaged over selected channels.
    Args:
        joints: as above; legs: as above.
        vcap:   positive scalar cap; k: slope of the logistic.
    """
    def __init__(self, joints=("hip_joint",), legs=None, vcap=2.0, k=8.0):
        super().__init__()
        self.joints = joints
        self.legs = ["FL","FR","RL","RR"] if legs is None else legs
        self.vcap = vcap
        self.k = k

    def forward(self, x, cond, time, *args):
        x = x[:, 1:, :]
        *_, qd, _, _ = split_obs(x)
        idxs = []
        for leg in self.legs:
            base = LEG_SLICES[leg].start
            for j in self.joints:
                idxs.append(base + JOINT_IN_LEG[j])
        idxs = torch.as_tensor(idxs, dtype=torch.long, device=qd.device)
        sel = qd.index_select(-1, idxs).abs()  # [B,H,len]
        r = torch.sigmoid(self.k * (self.vcap - sel)).mean(dim=-1, keepdim=True)
        return mean_over_time(r)
    
class ProjGravitySmoothRew(nn.Module):
    """
    Encourage temporal smoothness of the projected gravity vector g_proj (x[..., 18:21]).
    Modes:
        - 'l1'  : penalize ||g(t) - g(t-1)|| (L1 difference, default)
        - 'l2'  : penalize ||g(t) - g(t-1)||^2 (optionally after L2-normalization)
        - 'cos' : penalize (1 - cos(g(t), g(t-1)))^2 (direction-only, scale-invariant)
    Notes:
        - Drops the first timestep (x = x[:, 1:, :]) so no grad flows through t=0.
        - If normalize=True, vectors are L2-normalized before differencing.
    Output:
        reward : [batch, 1] (higher is better)
    """

    def __init__(self, mode: str = 'l1', normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        assert mode in ('l1', 'l2', 'cos')
        self.mode = mode
        self.normalize = normalize
        self.eps = eps

    def forward(self, x, cond=None, time=None, *args):
        # Ignore the first step for conditioning (no grad through t=0)
        x = x[:, 1:, :]

        # g_proj is the projected gravity in obs: x[..., 18:21]
        _, _, g_proj, _, _, _, _, _ = split_obs(x)  # [B, H-1, 3]

        # Optional normalization to focus on direction smoothness
        if self.normalize:
            g_norm = g_proj / (g_proj.norm(dim=-1, keepdim=True) + self.eps)
        else:
            g_norm = g_proj

        # If horizon is too short after dropping t=0, return zero reward
        if g_norm.size(1) < 2:
            return torch.zeros(g_norm.size(0), 1, device=g_norm.device, dtype=g_norm.dtype)

        if self.mode == 'cos':
            # Direction-only smoothness via cosine change
            g1 = g_norm[:, 1:, :]
            g0 = g_norm[:, :-1, :]
            cos = (g1 * g0).sum(dim=-1)     # [B, H-2]
            cos = torch.clamp(cos, -1.0, 1.0)
            pen = (1.0 - cos) ** 2
            pen = pen.unsqueeze(-1)         # [B, H-2, 1]
            return -mean_over_time(pen)     # → [B, 1]

        elif self.mode == 'l2':
            # L2 difference smoothness
            dg = g_norm[:, 1:, :] - g_norm[:, :-1, :]
            mse = (dg ** 2).mean(dim=-1, keepdim=True)  # [B, H-2, 1]
            return -mean_over_time(mse)                 # → [B, 1]

        else:  # self.mode == 'l1'
            # L1 difference smoothness
            dg = g_norm[:, 1:, :] - g_norm[:, :-1, :]
            mae = dg.abs().mean(dim=-1, keepdim=True)   # [B, H-2, 1]
            return -mean_over_time(mae)                 # → [B, 1]