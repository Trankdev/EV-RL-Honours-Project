"""
Running Mean and Std for normalization
来自 cMALC-D 项目
"""
import torch as th


class RunningMeanStd:
    """
    Tracks the mean, std and count of values using Welford's algorithm.
    """

    def __init__(self, epsilon=1e-4, shape=(), device="cpu"):
        """
        Args:
            epsilon: small value to avoid division by zero
            shape: shape of the data to track
            device: torch device
        """
        self.mean = th.zeros(shape, dtype=th.float32, device=device)
        self.var = th.ones(shape, dtype=th.float32, device=device)
        self.count = epsilon

    def update(self, x):
        """
        Updates the mean and variance using a batch of data.
        Flattens all dimensions so shape=() scalar stats are maintained.
        """
        x_flat = x.reshape(-1)  # flatten everything → 1D
        batch_mean = th.mean(x_flat)
        batch_var = th.var(x_flat, unbiased=False)
        batch_count = x_flat.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Updates from the mean, variance and count of a batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + th.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count