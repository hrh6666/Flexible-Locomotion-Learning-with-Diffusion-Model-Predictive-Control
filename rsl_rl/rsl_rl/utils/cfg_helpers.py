# Registry: w# ----------------------------- Argument coercion helpers -----------------------------
from typing import Any, Mapping, Sequence
import torch

_NUMERIC_KINDS = (int, float)

def _is_numeric_list(x: Any) -> bool:
    """Return True if x is a (possibly nested) flat list/tuple of only numbers."""
    if not isinstance(x, (list, tuple)):
        return False
    return all(isinstance(v, _NUMERIC_KINDS) for v in x)

def _to_tensor_maybe(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> Any:
    """
    Convert numeric list/tuple to torch.tensor on the given device.
    Keep scalars/bools/strings/dicts as-is. Never touch non-numeric lists (e.g. ['RL','RR']).
    """
    if _is_numeric_list(x):
        return torch.tensor(x, device=device, dtype=dtype)
    return x

def _coerce_args_recursively(args: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> Any:
    """
    Recursively walk a nested structure (dict/list/tuple) and:
      - convert any **numeric list/tuple** into torch.tensor
      - leave strings/bools/ints/floats as-is
      - leave non-numeric lists (e.g. ['RL','RR']) as-is
    This is generic enough for both AnalyticReward.* and Constraint.* constructors.
    """
    if isinstance(args, Mapping):
        return {k: _coerce_args_recursively(v, device, dtype) for k, v in args.items()}
    if isinstance(args, (list, tuple)):
        # If the whole list is numeric -> tensor; else recurse element-wise and keep list/tuple
        if _is_numeric_list(args):
            return torch.tensor(args, device=device, dtype=dtype)
        # Mixed or non-numeric -> keep container type, recurse children
        out = [ _coerce_args_recursively(v, device, dtype) for v in args ]
        return type(args)(out)
    # Base types: keep as-is
    return args

def instantiate_from_cfg(module, class_name: str, raw_args: Mapping[str, Any], device: torch.device):
    """
    Generic factory:
      - module: e.g. rsl_rl.modules.hand_crafted_rewards as AnalyticReward OR rsl_rl.modules.constraints as Constraint
      - class_name: string in that module
      - raw_args: plain-JSONable dict from cfg (numbers/strings/bools/lists/dicts only)
      - device: torch device, used for numeric-list -> torch.tensor conversion
    """
    cls = getattr(module, class_name)  # explicit failure if not found
    coerced = _coerce_args_recursively(raw_args, device=device, dtype=torch.float32)
    obj = cls(**coerced)
    # Most analytic rewards & constraints are nn.Module-like; calling .to() is safe if available
    if hasattr(obj, "to"):
        obj = obj.to(device)
    return obj
# ------------------------------------------------------------------------------------