"""PCA / t-SNE of the encoder latent z and the history-estimator latent ẑ of trained PrivilegedPPO checkpoints.

    .venv/bin/python docs/privileged_curriculum/latent_viz.py <out.png> <task>:<checkpoint.pt> [<task>:<checkpoint.pt> ...]

Rolls out each checkpoint (privileged values masked, deterministic actions) on episodes with randomized dynamics,
collects z = μ(e_t), ẑ = φ(history) and the true privileged values e_t, and plots per run:
PCA of z and ẑ coloured by mass, t-SNE of ẑ coloured by mass / second dynamics parameter / velocity, and the PCA
explained variance of z and ẑ (how many of the latent dimensions are used). Also prints held-out linear and kNN R²
from z and ẑ to each privileged group.

The checkpoints do not store the NormalizeObservation statistics, so they are re-estimated from warm-up episodes.
"""
import importlib.util
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "gym_privileged_ppo", os.path.join(HERE, "..", "..", "examples", "privileged_curriculum", "gym_privileged_ppo.py"))
ex = importlib.util.module_from_spec(spec)
sys.modules["gym_privileged_ppo"] = ex  # dataclasses look the module up by name
spec.loader.exec_module(ex)

from skrl.agents.torch.privileged_ppo import ObsLayout, PrivilegedGaussianPolicy  # noqa: E402


@torch.no_grad()
def collect(task_name: str, ckpt: str, warmup: int = 20, episodes: int = 40, seed: int = 123):
    task = ex.TASKS[task_name]
    env = ex.make_env(task_name, task, ex.Fixed(1.0), seed)
    n_vel, n_dyn = len(task.velocity_indices), len(task.dynamics)
    obs_dim = (env.observation_space.shape[0] - n_vel - n_dyn - 1) // (1 + task.history)
    layout = ObsLayout(obs_dim=obs_dim, history_length=task.history, privileged_dim=n_vel + n_dyn)
    policy = PrivilegedGaussianPolicy(env.observation_space, env.action_space, "cpu", layout=layout, hidden=128)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["policy"])
    low, high = torch.as_tensor(env.action_space.low), torch.as_tensor(env.action_space.high)
    z_all, zh_all, e_all = [], [], []
    for ep in range(warmup + episodes):
        if ep == warmup:
            ex._normalizer(env).update_running_mean = False
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32)[None]
            if ep >= warmup:
                z, z_hat = policy.trunk.latents(x)
                z_all.append(z[0].numpy()), zh_all.append(z_hat[0].numpy())
                e_all.append(layout.privileged(x)[0].numpy())
            action = policy.compute({"observations": x})[0][0].clamp(low, high).numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(z_all), np.array(zh_all), np.array(e_all), task


def r2(latent, target, model):
    a, b, ya, yb = train_test_split(latent, target, test_size=0.3, random_state=0)
    return r2_score(yb, model.fit(a, ya).predict(b))


def main() -> None:
    out, specs = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(len(specs), 6, figsize=(26, 4.2 * len(specs)), dpi=110, squeeze=False)
    rng = np.random.default_rng(0)
    for row, spec_ in enumerate(specs):
        task_name, ckpt = spec_.split(":", 1)
        z, zh, e, task = collect(task_name, ckpt)
        n_vel = len(task.velocity_indices)
        name = os.path.basename(ckpt).removesuffix(".pt")
        groups = {"velocities": list(range(n_vel))}
        groups.update({d: [n_vel + i] for i, d in enumerate(task.dynamics)})
        print(f"\n{name}: {len(z)} samples, latent dim {z.shape[1]}, privileged dim {e.shape[1]}")
        for lat_name, lat in (("z", z), ("ẑ", zh)):
            pca = PCA().fit(lat)
            pr = pca.explained_variance_.sum() ** 2 / (pca.explained_variance_ ** 2).sum()  # participation ratio
            scores = ", ".join(
                f"{g} lin {r2(lat, e[:, idx], RidgeCV()):.2f} / kNN {r2(lat, e[:, idx], KNeighborsRegressor(10)):.2f}"
                for g, idx in groups.items())
            print(f"  {lat_name}: PCA variance {np.round(pca.explained_variance_ratio_, 2)}, effective dim {pr:.1f}")
            print(f"  {lat_name} R²: {scores}")
        sub = rng.choice(len(z), size=min(3000, len(z)), replace=False)
        mass, dyn2, vel = e[sub, n_vel], e[sub, n_vel + 1], e[sub, 0]
        pz, pzh = PCA(2).fit_transform(z)[sub], PCA(2).fit_transform(zh)[sub]
        tsne = TSNE(2, perplexity=40, random_state=0, init="pca").fit_transform(zh[sub])
        panels = [
            (pz, mass, "PCA of z  (colour: mass)"),
            (pzh, mass, "PCA of ẑ  (colour: mass)"),
            (tsne, mass, "t-SNE of ẑ  (colour: mass)"),
            (tsne, dyn2, f"t-SNE of ẑ  (colour: {task.dynamics[1]})"),
            (tsne, vel, "t-SNE of ẑ  (colour: velocity 0)"),
        ]
        for col, (xy, c, title) in enumerate(panels):
            ax = axes[row][col]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=3, cmap="viridis")
            fig.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
        axes[row][0].set_ylabel(name, fontsize=10)
        ax = axes[row][5]
        k = np.arange(1, z.shape[1] + 1)
        ax.bar(k - 0.2, PCA().fit(z).explained_variance_ratio_, 0.4, label="z")
        ax.bar(k + 0.2, PCA().fit(zh).explained_variance_ratio_, 0.4, label="ẑ")
        ax.set_title("PCA explained variance ratio", fontsize=10)
        ax.set_xlabel("component")
        ax.legend()
    fig.suptitle("PrivilegedPPO latents (8-dim): encoder z vs history estimator ẑ, rollouts with privileged masked")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
