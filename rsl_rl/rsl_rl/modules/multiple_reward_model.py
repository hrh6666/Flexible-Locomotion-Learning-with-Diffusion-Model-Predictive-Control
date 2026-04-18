import torch.nn as nn
import torch

class MultiRewardModel(nn.Module):
    """
    Wraps N independent reward models and returns their weighted sum.
    """
    def __init__(self, base_ctor, reward_names, weight_init=1.0, **ctor_kwargs):
        super().__init__()
        self.models = nn.ModuleDict({
            n: base_ctor(**ctor_kwargs) for n in reward_names
        })
        self.weights = torch.full((len(reward_names),), float(weight_init))
        self.names = reward_names
        
    def set_weights(self, weights):
        """
        Set the combination weights for each sub-model.

        Parameters
        ----------
        weights : Union[Sequence[float], Mapping[str, float]]
            If a sequence, must be length N in the same order as `self.names`.
            If a dict, keys must be a subset of `self.names`, values are floats.
        """
        # case 1: dict mapping names → weight
        if isinstance(weights, dict):
            for name, w in weights.items():
                if name not in self.names:
                    raise KeyError(f"Unknown reward name: {name}")
                idx = self.names.index(name)
                self.weights[idx] = float(w)
        else:
            # assume a list/tuple of floats
            seq = list(weights)
            if len(seq) != len(self.names):
                raise ValueError(f"Expected {len(self.names)} weights, got {len(seq)}")
            # overwrite all at once
            new_w = torch.tensor(seq, device=self.weights.device, dtype=self.weights.dtype)
            self.weights[:] = new_w

    def forward(self, *args, **kwargs):
        """
        Return:
          combined_pred : Tensor[B]   Total reward
        """
        preds = []
        for i, n in enumerate(self.names):
            if self.weights[i] == 0:
                continue
            p = self.models[n](*args, **kwargs)
            preds.append(self.weights[i] * p)
        return sum(preds)
    
    def predict_components(self, *args, **kwargs):
        """
        Compute and return individual predictions for each reward model.

        Parameters
        ----------
        *args, **kwargs : passed through to each sub-model (same as forward).

        Returns
        -------
        pred_dict : dict[str, Tensor]
            A mapping from each reward name to its raw prediction Tensor[B].
        """
        # Loop over each named sub-model and collect its output
        return {
            name: model(*args, **kwargs)
            for name, model in self.models.items()
        }
    
    

    # diffusion-style loss helper，for .p_losses(...)
    def p_losses(self, x0, cond, target_dict, t):
        loss, loss_dict = 0., {}
        for n in self.names:
            l, _ = self.models[n].p_losses(
                x0, cond, target_dict[n], t
            )
            loss += l
            loss_dict[n] = l
        return loss, loss_dict