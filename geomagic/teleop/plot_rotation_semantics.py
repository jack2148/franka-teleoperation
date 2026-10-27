"""Plot saved offline traces with the existing system plotting environment."""
from pathlib import Path
import sys
import numpy as np


def plot(folder):
    folder = Path(folder)
    with np.load(folder / "step_hold_traces.npz") as traces:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
        for ax, name in zip(axes, ("step_30_x_500", "ramp_30_x_deadband_True")):
            a = traces[name]
            for col, label in ((1, "Raw mapped"), (2, "Deadband"), (3, "Conditioned"), (4, "Actual")):
                ax.plot(a[:, 0], a[:, col], label=label)
            ax.set(title=name, xlabel="Simulation time (s)", ylabel="Relative orientation (deg)")
            ax.grid(alpha=.3)
            ax.legend()
        fig.savefig(folder / "step_hold.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    plot(sys.argv[1])
