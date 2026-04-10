"""Active Units monitoring for adaptive latent dimension discovery.

Active Units (AU) counts how many latent dimensions have
Var_x[E[z_i|x]] > threshold — i.e., dimensions the encoder
actually uses to encode input-dependent information.
"""


class ActiveUnitsMonitor:
    """Track AU readings across epochs and detect stabilization.

    Stabilization means the last ``stability_count`` AU readings vary
    by at most 1 (max - min <= 1).

    Args:
        check_interval:  Compute AU every N epochs.
        stability_count: How many consecutive readings must be stable.
    """

    def __init__(self, check_interval, stability_count):
        self.check_interval = check_interval
        self.stability_count = stability_count
        self.au_history = []
        self._disabled = False

    @property
    def latest(self):
        """Most recent AU reading."""
        return self.au_history[-1]

    def should_check(self, epoch, warmup):
        """Return True if AU should be computed this epoch."""
        if self._disabled:
            return False
        if epoch <= warmup:
            return False
        return epoch % self.check_interval == 0

    def record(self, active_count):
        """Append an AU reading."""
        self.au_history.append(active_count)

    def is_stable(self):
        """True if the last ``stability_count`` readings differ by at most 1."""
        if len(self.au_history) < self.stability_count:
            return False
        window = self.au_history[-self.stability_count:]
        return max(window) - min(window) <= 1

    def disable(self):
        """Permanently stop monitoring."""
        self._disabled = True

    def get_result(self, original_z1_size, epoch, user_accepted):
        """Build the JSON-serializable result dict."""
        return {
            "discovered_z1_size": self.latest,
            "original_z1_size": original_z1_size,
            "stable_at_epoch": epoch,
            "au_history": list(self.au_history),
            "user_accepted": user_accepted,
        }
