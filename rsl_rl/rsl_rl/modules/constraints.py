# rsl_rl/modules/constraints.py
# Extensible & composable constraint system for diffusion sampling.
# IMPORTANT: All constraints here operate ONLY on timesteps t >= 1
# to preserve conditioning consistency at t=0.
# rsl_rl/modules/constraints.py
# Extensible & composable constraint system for diffusion sampling.
# IMPORTANT:
#   - All constraints here operate ONLY on timesteps t >= 1 (never touch t=0).
#   - Name-to-index resolution uses built-in defaults (ACTION_DIM/OBS_DIM layout),
#     so you DON'T need to build or inject named indices anywhere else.

from __future__ import annotations
import torch
from typing import List, Sequence, Union, Optional, Dict, Protocol

Tensor = torch.Tensor

# ---------------------------------------------------------------------
# Built-in trajectory layout (match your hand-crafted reward file)
# ---------------------------------------------------------------------
ACTION_DIM = 12
OBS_DIM    = 49
OBS_OFF    = ACTION_DIM  # global offset for all obs slices within the transition vector

def split_obs(x: Tensor):
    """
    Split the *trajectory* vector into named OBS slices with a +12 shift for actions.
    x: [B, H, T] where the first 12 dims are the current action (not returned),
       followed by 49-D observation laid out as documented in your repo.
    Returns (each [B, H, *]):
        v_lin, v_ang, g_proj, cmd, q, qd, u_prev, mu
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

# Precomputed default name→indices mapping (module-level, used automatically)
_DEFAULT_NAMED: Dict[str, Sequence[int]] = {
    "actions": list(range(0, ACTION_DIM)),
    "obs_v_lin": list(range(OBS_OFF + 0,  OBS_OFF + 3)),
    "obs_v_ang": list(range(OBS_OFF + 3,  OBS_OFF + 6)),
    "obs_g":     list(range(OBS_OFF + 6,  OBS_OFF + 9)),
    "obs_cmd":   list(range(OBS_OFF + 9,  OBS_OFF +12)),
    "obs_q":     list(range(OBS_OFF +12, OBS_OFF +24)),
    "obs_qd":    list(range(OBS_OFF +24, OBS_OFF +36)),
    "obs_u_prev":list(range(OBS_OFF +36, OBS_OFF +48)),
    "obs_mu":    list(range(OBS_OFF +48, OBS_OFF +49)),
}


# --- Per-leg index helpers (copy if not defined yet in this file) ---
LEG_SLICES = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}
JOINT_IN_LEG = {"hip_joint": 0, "thigh_joint": 1, "calf_joint": 2}

def _as_leg_list(leg_or_legs):
    """Accept a single leg 'FL' or a list/tuple; always return a non-empty list of legs."""
    if isinstance(leg_or_legs, (list, tuple)):
        assert len(leg_or_legs) > 0, "legs list must be non-empty"
        return list(leg_or_legs)
    return [leg_or_legs]

# ---------------------------------------------------------------------
# Constraint core
# ---------------------------------------------------------------------
class ConstraintContext:
    """
    Runtime context available to constraints. Other files do NOT need to provide named_indices.
    If ctx.named_indices is None or missing a name, we FALL BACK to the built-in _DEFAULT_NAMED.
    """
    def __init__(
        self,
        transition_dim: int,
        action_dim: int,
        named_indices: Optional[Dict[str, Sequence[int]]] = None,
    ):
        self.transition_dim = int(transition_dim)
        self.action_dim = int(action_dim)
        self.named_indices = named_indices  # optional

    def idx(self, name_or_indices: Union[str, Sequence[int]]) -> torch.LongTensor:
        """
        Resolve a name to indices using:
          1) ctx.named_indices (if provided and contains the name)
          2) module-level _DEFAULT_NAMED (built from ACTION_DIM/OBS layout)
        Or accept a raw sequence of ints.
        """
        if isinstance(name_or_indices, str):
            # try user-provided mapping first (if any)
            if self.named_indices is not None and name_or_indices in self.named_indices:
                arr = self.named_indices[name_or_indices]
                return torch.as_tensor(arr, dtype=torch.long)
            # fallback to built-in mapping
            if name_or_indices in _DEFAULT_NAMED:
                return torch.as_tensor(_DEFAULT_NAMED[name_or_indices], dtype=torch.long)
            raise KeyError(f"Unknown index name: {name_or_indices}. "
                           f"Known default keys: {list(_DEFAULT_NAMED.keys())}")
        else:
            return torch.as_tensor(name_or_indices, dtype=torch.long)


class Constraint(Protocol):
    """Interface for any constraint in the chain."""
    def apply(self, x: Tensor, ctx: ConstraintContext) -> Tensor: ...


class ConstraintChain:
    """
    Sequential composer: applies constraints in the order they were added.
    Order matters; later constraints can refine earlier ones.
    """
    def __init__(self, constraints: Optional[List[Constraint]] = None):
        self._constraints: List[Constraint] = list(constraints or [])

    def add(self, c: Constraint) -> "ConstraintChain":
        self._constraints.append(c)
        return self

    @torch.no_grad()
    def __call__(self, x: Tensor, ctx: ConstraintContext) -> Tensor:
        # x: [B, H, T]; all constraints work only on x[:, 1:, :]
        for c in self._constraints:
            x = c.apply(x, ctx)
        return x

# ---------------- Built-in constraints (operate on t >= 1) ---------------- #

class BoxConstraint:
    """
    Per-dimension box clamp on selected coordinates for timesteps t >= 1.
    Args:
      indices: name (e.g., "actions", "obs_q") or raw index sequence into last dim.
      low/high: scalar, Tensor[K], or broadcastable onto [..., K].
    """
    def __init__(self, indices: Union[str, Sequence[int]], low: Union[float, Tensor], high: Union[float, Tensor]):
        self.indices = indices
        self.low = low
        self.high = high

    @torch.no_grad()
    def apply(self, x: Tensor, ctx: ConstraintContext) -> Tensor:
        if x.size(1) < 2:
            return x  # nothing beyond t=0
        idx = ctx.idx(self.indices).to(device=x.device)
        sl = x[:, 1:, :][..., idx]  # [B, H-1, K]
        low = self.low if torch.is_tensor(self.low) else torch.as_tensor(self.low, device=x.device, dtype=x.dtype)
        high = self.high if torch.is_tensor(self.high) else torch.as_tensor(self.high, device=x.device, dtype=x.dtype)
        low = low.to(x).view(*([1] * (sl.dim() - 1)), -1)
        high = high.to(x).view(*([1] * (sl.dim() - 1)), -1)
        # print("sl",sl)
        # before = sl.clone()              # save copy
        sl.clamp_(min=low, max=high)     # in-place clamp
        # print("Diff:", (sl - before))    # print difference
        # print("Box Constraint Activated", "Low", self.low, "High", self.high)
        x[:, 1:, :][..., idx] = sl
        return x


class RateLimitConstraint:
    """
    Per-step rate limit for selected coordinates on t >= 1:
        Δx_t = x_t - x_{t-1}, clamp Δ to [-max_delta, +max_delta].
    Implemented with parallel tensor ops (no Python loop).
    """
    def __init__(self, indices: Union[str, Sequence[int]], max_delta: Union[float, Tensor]):
        self.indices = indices
        self.max_delta = max_delta

    @torch.no_grad()
    def apply(self, x: Tensor, ctx: ConstraintContext) -> Tensor:
        if x.size(1) < 2:
            return x
        idx = ctx.idx(self.indices).to(device=x.device)
        y = x[..., idx]  # [B, H, K]

        # compute deltas
        dy = y[:, 1:, :] - y[:, :-1, :]  # [B, H-1, K]

        # broadcast max_delta
        md = self.max_delta if torch.is_tensor(self.max_delta) else \
             torch.as_tensor(self.max_delta, device=x.device, dtype=x.dtype)
        md = md.view(*([1] * (dy.dim() - 1)), -1)

        # clamp deltas in one shot
        dy = dy.clamp(min=-md, max=md)

        # reconstruct trajectory: y0 + cumsum(dy)
        y_new = torch.cat([y[:, :1, :], y[:, :1, :] + torch.cumsum(dy, dim=1)], dim=1)

        # replace in x
        x[..., idx] = y_new
        return x
    
class LegBoxConstraint:
    """
    Hard box projection for selected legs & joints.

    Args:
        legs   : str or List[str], e.g. "FL" or ["FL","FR","RL","RR"].
        joints : tuple/list like ("hip_joint","thigh_joint","calf_joint") or a subset.
        space  : str in {"obs_q","obs_qd","actions"} specifying which 12-d block to clamp.
                 - "actions": first 12 dims of the transition vector (index 0..11)
                 - "obs_q" : observation joint positions (offset ACTION_DIM + 12 .. +24)
                 - "obs_qd": observation joint velocities (offset ACTION_DIM + 24 .. +36)
        lo, hi : bound specs. Each can be:
                 - scalar (same bound for all selected channels),
                 - dict by joint name, e.g. lo={"hip_joint":-0.2, "thigh_joint":-0.5},
                 - dict by (leg, joint), e.g. hi={("FL","hip_joint"):0.4, ("FR","hip_joint"):0.35},
                 - nested dict by leg -> joint, e.g. lo={"FL":{"hip_joint":-0.25}}
                 Resolution priority (most specific first):
                     (leg, joint)  >  leg->{joint}  >  {joint}  >  scalar default
    """

    def __init__(
        self,
        legs,
        joints=("hip_joint","thigh_joint","calf_joint"),
        space: str = "obs_q",
        lo=-1.0,
        hi= 1.0,
    ):
        self.legs = _as_leg_list(legs)
        self.joints = list(joints)
        assert space in ("obs_q","obs_qd","actions"), "space must be one of {'obs_q','obs_qd','actions'}"
        self.space = space
        self.lo_spec = lo
        self.hi_spec = hi

    @staticmethod
    def _resolve_bound(spec, leg, joint, default):
        """Return a float bound for (leg, joint) from flexible 'spec' (scalar/tensor/dict...)."""
        # scalar
        if isinstance(spec, (int, float)):
            return float(spec)
        # 0-dim tensor
        if torch.is_tensor(spec) and spec.dim() == 0:
            return float(spec.item())
        # dict-like resolution
        if isinstance(spec, dict):
            # 1) pair key (leg, joint)
            if (leg, joint) in spec:
                v = spec[(leg, joint)]
                return float(v if not torch.is_tensor(v) else v.item())
            # 2) nested: spec[leg][joint]
            if leg in spec and isinstance(spec[leg], dict) and (joint in spec[leg]):
                v = spec[leg][joint]
                return float(v if not torch.is_tensor(v) else v.item())
            # 3) by joint only
            if joint in spec:
                v = spec[joint]
                return float(v if not torch.is_tensor(v) else v.item())
        # fallback
        return float(default)

    def _space_base_offset(self) -> int:
        """Return the starting offset of the 12-d block (actions or a 12-d obs slice)."""
        if self.space == "actions":
            return 0
        elif self.space == "obs_q":
            return OBS_OFF + 12  # q block
        elif self.space == "obs_qd":
            return OBS_OFF + 24  # qd block
        raise RuntimeError("invalid space")

    @torch.no_grad()
    def apply(self, x: Tensor, ctx: ConstraintContext) -> Tensor:
        # Do nothing if horizon < 2 (we never modify t=0)
        if x.size(1) < 2:
            return x

        base = self._space_base_offset()

        # Build channel indices and the corresponding per-channel bounds
        idxs, lo_list, hi_list = [], [], []
        for leg in self.legs:
            leg_base = base + LEG_SLICES[leg].start  # start index for this leg within the 12-d block
            for j in self.joints:
                ch_idx = leg_base + JOINT_IN_LEG[j]
                idxs.append(ch_idx)
                lo_val = self._resolve_bound(self.lo_spec, leg, j, default=-1.0)
                hi_val = self._resolve_bound(self.hi_spec, leg, j, default= 1.0)
                # ensure lo <= hi to avoid inverted bounds
                if lo_val > hi_val:
                    lo_val, hi_val = hi_val, lo_val
                lo_list.append(lo_val)
                hi_list.append(hi_val)

        idxs = torch.as_tensor(idxs, dtype=torch.long, device=x.device)

        # Slice out the selected channels on t >= 1 -> shape [B, H-1, C]
        sl = x[:, 1:, :][..., idxs]

        # Prepare broadcastable bound tensors, shape [1,1,C]
        lo_t = torch.tensor(lo_list, dtype=x.dtype, device=x.device).view(1,1,-1)
        hi_t = torch.tensor(hi_list, dtype=x.dtype, device=x.device).view(1,1,-1)

        # In-place clamp (hard projection)
        sl.clamp_(min=lo_t, max=hi_t)

        # Write back
        x[:, 1:, :][..., idxs] = sl
        return x


# ---------------------------------------------------------------------
# (Optional) keep a helper for external users who still want a dict
# ---------------------------------------------------------------------
def build_named_indices(action_dim: int = ACTION_DIM, observation_dim: int = OBS_DIM) -> Dict[str, Sequence[int]]:
    """
    Provided for API compatibility. Returns the same mapping as _DEFAULT_NAMED
    using the module-level ACTION_DIM/OBS_DIM/OBS_OFF.
    """
    # We ignore the arguments and return the defaults, to avoid "building" elsewhere.
    return dict(_DEFAULT_NAMED)