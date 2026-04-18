import math
from typing import Dict

def cosine_schedule(iteration, base_scale, params):
    period = params["period"]
    return base_scale * 0.5 * (1 + math.cos(math.pi * iteration / period))


def exp_cycle_schedule(it: int,
                       base_scale: float,
                       params: Dict) -> float:
    """
    Periodic exponential-decay schedule with floor truncation.

    Args
    ----
    it : int
        Current outer iteration number (0-based).
    base_scale : float
        Initial noise standard deviation at the *beginning of each cycle*.
    params : dict
        Hyper-parameters expected in cfg["policy"]["action_noise"]["schedule_params"]:
            decay_rate : float, optional (default=0.99)
                Multiplicative decay factor per iteration, e.g. 0.99.
            floor : float, optional (default=0.1)
                Minimal noise scale allowed after clipping.
            period : int, optional (default=100)
                Cycle length (iterations).  
                After every `period` steps the schedule **restarts** from `base_scale`.

    Returns
    -------
    float
        Noise scale for the current iteration.
    """
    # --- read params with sensible defaults ---------------------------
    decay  = params["decay_rate"]
    floor  = params["floor"]
    period = params["period"]

    # --- position inside the current cycle ----------------------------
    cycle_it = it % period            # avoid div-by-zero
    scale    = base_scale * (decay ** cycle_it)

    # --- truncate by floor -------------------------------------------
    return max(scale, floor)