import torch

class AmpRolloutStorage:
    """
    Tiny on-policy buffer for AMP.
    Stores the concatenated transition (state_size*2) every env-step.
    Shape: [T, N, D]
    """
    def __init__(self, num_envs, horizon, trans_dim, device):
        self.device   = device
        self.horizon  = horizon
        self.num_envs = num_envs
        self.buf      = torch.zeros(horizon, num_envs, trans_dim, device=device)
        self.step     = 0

    @torch.no_grad()
    def add(self, trans):                # trans: [N, D]
        self.buf[self.step].copy_(trans)
        self.step += 1
        assert self.step <= self.horizon, "AmpRolloutStorage overflow"

    def mini_batches(self, batch, epochs):
        total = self.step * self.num_envs
        flat  = self.buf[:self.step].reshape(total, -1)
        idx   = torch.arange(total, device=self.device)
        for _ in range(epochs):
            perm = idx[torch.randperm(total)]
            for s in range(0, total, batch):
                yield flat[perm[s:s+batch]]

    def clear(self):  
        self.step = 0