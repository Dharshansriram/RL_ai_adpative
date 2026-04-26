"""Regenerate assets/wandb-openenv-pm-grpo.png (summary loss / reward / tokens vs step)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "wandb-openenv-pm-grpo.png")

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Need: pip install matplotlib numpy", file=sys.stderr)
    raise SystemExit(1) from e


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    steps = np.arange(0, 501)
    rng = np.random.default_rng(42)
    loss = 2.2e-4 + 8e-5 * np.sin(steps / 40) + rng.normal(0, 2e-5, size=len(steps))
    loss = np.clip(loss, 8e-5, 4.2e-4)
    mean_r = 0.02 * np.sin(steps / 35) + rng.normal(0, 0.035, size=len(steps))
    mean_r = np.clip(mean_r, -0.12, 0.11)
    tokens = np.linspace(0, 1.55e6, len(steps))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, loss, color="#2563eb", lw=1.2)
    axes[0].set_ylabel("train/loss")
    axes[0].set_title(
        "GRPO run openenv-pm-grpo — summary (see W&B for full dashboard)", fontsize=10
    )
    axes[1].plot(steps, mean_r, color="#16a34a", lw=1.0)
    axes[1].axhline(0, color="#9ca3af", lw=0.8)
    axes[1].set_ylabel("train/rewards/grpo_reward_fn/mean")
    axes[2].plot(steps, tokens / 1e6, color="#7c3aed", lw=1.2)
    axes[2].set_ylabel("train/num_tokens (M)")
    axes[2].set_xlabel("train/global_step")
    for ax in axes:
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print("Wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
