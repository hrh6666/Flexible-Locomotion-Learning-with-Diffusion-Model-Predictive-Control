import numpy as np
import scipy.interpolate as interpolate


class Normalizer:
    """
    Base Normalizer class.
    Subclasses should implement the `normalize` and `unnormalize` methods.
    """
    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return f"[ Normalizer ] dim: {self.mins.size}\n    min: {np.round(self.mins, 2)}\n    max: {np.round(self.maxs, 2)}\n"

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the normalize method.")

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the unnormalize method.")

class DebugNormalizer(Normalizer):
    """
    Debug Normalizer – Identity mapping.
    Useful when no normalization is desired.
    """
    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x

class GaussianNormalizer(Normalizer):
    """
    Normalizes data to have zero mean and unit variance (Z-score normalization).
    """
    def __init__(self, X):
        super().__init__(X)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1  # scale factor if needed

    def __repr__(self):
        return f"[ GaussianNormalizer ] dim: {self.mins.size}\n    means: {np.round(self.means, 2)}\n    stds: {np.round(self.stds, 2)}\n"

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means

class LimitsNormalizer(Normalizer):
    """
    Linearly scales data from [min, max] to [-1, 1].
    """
    def normalize(self, x):
        # Scale data to [0, 1]
        x = (x - self.mins) / (self.maxs - self.mins)
        # Then scale to [-1, 1]
        return 2 * x - 1

    def unnormalize(self, x, eps=1e-4):
        # Clip to avoid out-of-bound values
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)
        # Scale back from [-1, 1] to [0, 1]
        x = (x + 1) / 2.
        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    """
    LimitsNormalizer that can handle constant dimensions.
    If a dimension is constant, a small epsilon is applied so that division by zero is avoided.
    """
    def __init__(self, X, eps=1):
        super().__init__(X)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f"[ SafeLimitsNormalizer ] Constant data in dimension {i} | min = max = {self.maxs[i]}")
                self.mins[i] -= eps
                self.maxs[i] += eps

class CDFNormalizer(Normalizer):
    """
    Transforms each feature using its empirical CDF so that the resulting distribution is uniform in [-1, 1].
    """
    def __init__(self, X):
        # Ensure the input is at least 2D
        X = atleast_2d(X)
        super().__init__(X)
        self.dim = self.X.shape[1]
        self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]

    def __repr__(self):
        cdf_str = " | ".join(f"{i}: {cdf}" for i, cdf in enumerate(self.cdfs))
        return f"[ CDFNormalizer ] dim: {self.mins.size}\n{cdf_str}"

    def wrap(self, fn_name, x):
        shape = x.shape
        # Reshape x to 2D for per-dimension processing
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    """
    Empirical CDF Normalizer for one dimension.
    Computes the empirical CDF and its inverse via interpolation.
    """
    def __init__(self, X):
        assert X.ndim == 1, "Input X must be 1D."
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob, bounds_error=False,
                                       fill_value=(cumprob[0], cumprob[-1]))
        self.inv = interpolate.interp1d(cumprob, quantiles, bounds_error=False,
                                        fill_value=(quantiles[0], quantiles[-1]))
        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return f"[{np.round(self.xmin, 2)}, {np.round(self.xmax, 2)}]"

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        # Compute the CDF value (in [0, 1])
        y = self.fn(x)
        # Map to [-1, 1]
        return 2 * y - 1

    def unnormalize(self, x, eps=1e-4):
        # Map from [-1, 1] back to [0, 1]
        x = (x + 1) / 2.
        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(f"[ CDFNormalizer1d ] Warning: value out of range during unnormalize: min {x.min()}, max {x.max()}")
        x = np.clip(x, self.ymin, self.ymax)
        return self.inv(x)

def empirical_cdf(sample):
    """
    Computes the empirical cumulative distribution function (CDF) for a 1D array.
    
    Returns:
       quantiles: Sorted unique values of the sample.
       cumprob: Cumulative probabilities corresponding to the quantiles.
    """
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

def atleast_2d(x):
    """
    Ensures that the input array is at least 2D.
    """
    if x.ndim < 2:
        x = x[:, None]
    return x

###########################################################
# Configurable Dataset Normalizer
###########################################################
class ConfigurableDatasetNormalizer:
    """
    A configurable dataset normalizer that wraps per-field normalizers.
    
    This class takes the complete dataset and a configuration dictionary.
    The configuration should include:
      - 'obs_normalizer': The string name of the normalizer class to use for observations 
                          (e.g., "LimitsNormalizer", "GaussianNormalizer", etc.).
      - 'action_normalizer': The string name of the normalizer class to use for actions 
                             (e.g., "DebugNormalizer", "GaussianNormalizer", etc.).
    
    The input dataset is expected to be a dictionary mapping field names to numpy arrays.
    
    Example usage:
       norm = ConfigurableDatasetNormalizer(dataset, cfg, path_lengths)
       normalized_obs = norm.normalize(obs, key="observations")
       original_obs = norm.unnormalize(normalized_obs, key="observations")
    """
    def __init__(self, dataset, norm_str):
        # If path_lengths is provided, flatten the dataset trajectory data; otherwise assume dataset is pre-flattened.
        self.dataset = dataset
        self.normalizers = {}
        # Initialize a per-field normalizer according to the key
        for key, val in self.dataset.items():
            try:
                norm_class = eval(norm_str)
                print(key, val.shape)
                self.normalizers[key] = norm_class(val)
            except Exception as e:
                print(f"[ ConfigurableDatasetNormalizer ] Skipping key '{key}' due to error: {e}")

    def normalize(self, x, key):
        """
        Normalizes input x using the normalizer for the given key.
        """
        if key in self.normalizers:
            return self.normalizers[key].normalize(x)
        else:
            raise KeyError(f"Key '{key}' not found in normalizers.")

    def unnormalize(self, x, key):
        """
        Unnormalizes input x using the normalizer for the given key.
        """
        if key in self.normalizers:
            return self.normalizers[key].unnormalize(x)
        else:
            raise KeyError(f"Key '{key}' not found in normalizers.")

    def get_field_normalizers(self):
        """
        Returns the dictionary mapping field names to their corresponding normalizer objects.
        """
        return self.normalizers

    def __call__(self, x, key):
        return self.normalize(x, key)