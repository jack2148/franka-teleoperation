"""Bounded-memory timing summaries; never subtract clocks from different PCs."""
from array import array
import math


class TimingStats:
    """Exact count/mean/max; exact quantiles until the explicit sample cap.

    If exceeded, quantiles describe the retained prefix and are marked truncated.
    All observations still contribute to count, mean and maximum.
    """
    def __init__(self, capacity=100000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.values = array("d")
        self.count = 0
        self.total = 0.
        self.maximum = None

    def add(self, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("timing must be finite and nonnegative")
        self.count += 1
        self.total += value
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.values) < self.capacity:
            self.values.append(value)

    def report(self):
        ordered = sorted(self.values)

        def percentile(q):
            if not ordered:
                return None
            index = (len(ordered) - 1) * q
            low = int(index)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

        return dict(count=self.count, mean=self.total / self.count if self.count else None,
                    median=percentile(.5), p95=percentile(.95), p99=percentile(.99),
                    maximum=self.maximum, quantile_samples=len(ordered),
                    quantiles_truncated=self.count > len(ordered))


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("value must be finite and positive")
    return number
