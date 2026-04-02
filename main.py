
"""
Comprehensive comparison of continuous action actor-critic algorithms (PPO, SAC, TD3)
on Gymnasium MuJoCo tasks with detailed analysis.

Usage (default arguments shown):

    python rl_compare_fixed.py \
        --env HalfCheetah-v5 \
        --timesteps 200000 \
        --eval_episodes 10 \
        --seeds 42 123 456 \
        --output results \
        --algos PPO SAC TD3

Requirements:
    pip install gymnasium[mujoco] stable-baselines3[extra] matplotlib pandas seaborn scipy psutil
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import seaborn as sns

sns.set_theme(context="talk", style="darkgrid")

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}

ON_POLICY = {"PPO"}

# ---------------------------------------------------------------------------
# Hyperparameters — RL-Zoo tuned defaults for HalfCheetah-v5
# Source: https://github.com/DLR-RM/rl-baselines3-zoo
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS: dict[str, dict] = {
    "PPO": dict(
        n_steps=512,
        batch_size=64,
        gamma=0.98,
        learning_rate=2.0633e-05,
        ent_coef=0.000401762,
        clip_range=0.1,
        n_epochs=20,
        gae_lambda=0.92,
        max_grad_norm=0.8,
        vf_coef=0.58096,
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        device="cpu",
    ),
    "SAC": dict(
        learning_starts=10_000,
        ent_coef="auto",
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        train_freq=1,
        gradient_steps=1,
    ),
    "TD3": dict(
        learning_rate=1e-3,
        batch_size=256,
        gamma=0.99,
        buffer_size=1_000_000,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
}

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(env_id: str, max_episode_steps: int = 1000):
    """Return a callable that builds a monitored, time-limited environment."""

    def _init():
        env = gym.make(env_id)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return Monitor(env)

    return _init


def build_train_env(env_id: str) -> DummyVecEnv:
    """Build training env, applying VecNormalize only for on-policy algorithms."""
    venv = DummyVecEnv([make_env(env_id)])
    return venv


def build_eval_env(env_id: str) -> DummyVecEnv:
    """
    Build a separate evaluation env.
    """
    venv = DummyVecEnv([make_env(env_id)])
    return venv


class LearningCurveCallback(BaseCallback):
    """
    Evaluates the policy every `eval_freq` steps, records the mean reward,
    and saves the model whenever a new best is achieved.
    Replaces the combination of EvalCallback + a separate tracking callback.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 5,
        best_model_save_path: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.evaluations_timesteps: List[int] = []
        self.evaluations_results: List[float] = []
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(mean_reward)

            if self.verbose > 0:
                print(f"  [eval] t={self.num_timesteps:,}  reward={mean_reward:.2f} ± {std_reward:.2f}")

            if mean_reward > self._best_mean_reward and self.best_model_save_path:
                self._best_mean_reward = mean_reward
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.verbose > 0:
                    print(f"  [eval] New best model saved ({mean_reward:.2f})")

        return True

def evaluate_policy_extended(model, env, n_episodes: int) -> dict:
    """Deterministic rollout; returns reward statistics and mean episode length."""
    rewards, lengths = [], []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
    }

def record_video(
    model,
    algo_name: str,
    env_id: str,
    out_dir: Path,
    seed: int,
    n_episodes: int = 1,
):
    """Record a video of the trained policy and save as .mp4."""
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # RecordVideo requires a non-vectorized, non-monitored env
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=f"{algo_name}_seed{seed}",
        episode_trigger=lambda ep: True,  # record every episode
    )

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            # model.predict expects a batched obs for VecEnv-trained models
            action, _ = model.predict(obs[np.newaxis], deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated

    env.close()
    print(f"  [video] Saved to {video_dir}/")

# ---------------------------------------------------------------------------
# Train + evaluate one (algo, seed) combination
# ---------------------------------------------------------------------------
def train_and_evaluate(
    algo_name: str,
    env_id: str,
    timesteps: int,
    eval_episodes: int,
    seed: int,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2  # MB
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_env = build_train_env(env_id)
    eval_env = build_eval_env(env_id)

    model_cls = ALGORITHMS[algo_name]
    hyperparams = DEFAULT_HYPERPARAMS[algo_name].copy()

    model = model_cls(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
        **hyperparams,
    )

    callback = LearningCurveCallback(
        eval_env,
        eval_freq=5_000,
        n_eval_episodes=5,
        best_model_save_path=str(out_dir),
        verbose=1,
    )

    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    train_time = time.time() - start

    model.save(out_dir / f"{algo_name.lower()}_final")

    # Final evaluation
    metrics = evaluate_policy_extended(model, eval_env, n_episodes=eval_episodes)

    # Memory (indicative — not subprocess-isolated;)
    mem_after = process.memory_info().rss / 1024**2
    metrics["peak_memory_mb"] = mem_after - mem_before
    if TORCH_AVAILABLE and torch.cuda.is_available():
        metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    metrics.update(
        {
            "algo": algo_name,
            "seed": seed,
            "train_time": float(train_time),
            "timesteps": timesteps,
            "learning_curve_timesteps": callback.evaluations_timesteps,
            "learning_curve_rewards": callback.evaluations_results,
        }
    )

    # Record video of final policy
    record_video(model, algo_name, env_id, out_dir, seed)

    train_env.close()
    eval_env.close()
    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(all_results: List[Dict]):
    results_without_curves = []
    for r in all_results:
        r_copy = r.copy()
        r_copy.pop("learning_curve_timesteps", None)
        r_copy.pop("learning_curve_rewards", None)
        results_without_curves.append(r_copy)

    df = pd.DataFrame(results_without_curves)
    grouped = df.groupby("algo")

    summary = []
    for algo, algo_data in grouped:
        summary.append(
            {
                "algo": algo,
                "mean_reward": algo_data["mean_reward"].mean(),
                "mean_reward_std": algo_data["mean_reward"].std(),
                "median_reward": algo_data["median_reward"].mean(),
                "min_reward": algo_data["min_reward"].mean(),
                "max_reward": algo_data["max_reward"].mean(),
                "train_time_mean": algo_data["train_time"].mean(),
                "train_time_std": algo_data["train_time"].std(),
                "mean_episode_length": algo_data["mean_episode_length"].mean(),
                "peak_memory_mb": algo_data["peak_memory_mb"].mean(),
            }
        )

    return pd.DataFrame(summary), df

def statistical_tests(detailed_df: pd.DataFrame, out_dir: Path):
    """
    Pairwise Mann-Whitney U tests (non-parametric, appropriate for n=3 seeds).
    Results flagged at p < 0.05 are labelled as indicative given the small sample.
    """
    algos = detailed_df["algo"].unique()
    lines = [
        "Statistical Significance Tests — Mann-Whitney U (non-parametric)\n",
        "NOTE: With only 3 seeds per algorithm the test has low power.\n",
        "Results should be treated as indicative, not conclusive.\n",
        "=" * 65 + "\n\n",
    ]

    for i, algo1 in enumerate(algos):
        for algo2 in algos[i + 1 :]:
            r1 = detailed_df[detailed_df["algo"] == algo1]["mean_reward"].values
            r2 = detailed_df[detailed_df["algo"] == algo2]["mean_reward"].values

            stat, p = stats.mannwhitneyu(r1, r2, alternative="two-sided")

            lines.append(f"{algo1} vs {algo2}:\n")
            lines.append(f"  U-statistic : {stat:.4f}\n")
            lines.append(f"  p-value     : {p:.4f}\n")

            if p < 0.05:
                better = algo1 if np.mean(r1) > np.mean(r2) else algo2
                lines.append(f"  Result      : {better} appears better (p < 0.05, indicative)\n")
            else:
                lines.append(f"  Result      : No significant difference (p >= 0.05)\n")
            lines.append("\n")

    text = "".join(lines)
    (out_dir / "statistical_tests.txt").write_text(text)
    print("\n" + text)


def plot_results(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    all_results: List[Dict],
    out_dir: Path,
    total_timesteps: int,
):
    # 1. Mean reward bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(
        summary_df["algo"],
        summary_df["mean_reward"],
        yerr=summary_df["mean_reward_std"],
        capsize=10,
        alpha=0.8,
        edgecolor="black",
    )
    plt.title("Mean Episode Reward (± std across seeds)")
    plt.ylabel("Reward")
    plt.xlabel("Algorithm")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_reward.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(
        summary_df["algo"],
        summary_df["train_time_mean"],
        yerr=summary_df["train_time_std"],
        capsize=10,
        alpha=0.8,
        edgecolor="black",
    )
    plt.title("Training Time (wall clock)")
    plt.ylabel("Seconds")
    plt.xlabel("Algorithm")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_time.png", dpi=300)
    plt.close()

    common_grid = np.linspace(0, total_timesteps, 60)
    plt.figure(figsize=(12, 7))

    for algo in summary_df["algo"].unique():
        algo_results = [r for r in all_results if r["algo"] == algo]
        interpolated_curves = []

        for r in algo_results:
            ts = np.array(r.get("learning_curve_timesteps", []))
            rw = np.array(r.get("learning_curve_rewards", []))
            if len(ts) < 2:
                continue
            f = interp1d(ts, rw, bounds_error=False, fill_value=(rw[0], rw[-1]))
            interpolated_curves.append(f(common_grid))

        if not interpolated_curves:
            continue

        curves = np.array(interpolated_curves)          # shape: (n_seeds, n_points)
        mean_r = curves.mean(axis=0)
        std_r = curves.std(axis=0)

        plt.plot(common_grid, mean_r, label=algo, linewidth=2)
        plt.fill_between(common_grid, mean_r - std_r, mean_r + std_r, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title("Learning Curves (mean ± std across seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    detailed_df.boxplot(column="mean_reward", by="algo", grid=False)
    plt.suptitle("")
    plt.title("Reward Distribution Across Seeds")
    plt.ylabel("Mean Reward")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_distribution.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def save_config(args, out_dir: Path):
    config = {
        "env": args.env,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "seeds": args.seeds,
        "algorithms": args.algos,
        "hyperparameters": DEFAULT_HYPERPARAMS,
        "normalization": {
            "on_policy": "VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10)",
            "off_policy": "none (replay buffer stationarity)",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": os.sys.version,
    }
    with open(out_dir / "config.json", "w") as f:
        # Some hyperparameters (e.g., torch activation classes) are not JSON-native.
        json.dump(config, f, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of PPO, SAC, TD3 on MuJoCo tasks."
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        default=["PPO", "SAC", "TD3"],
        choices=list(ALGORITHMS.keys()),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, out_dir)

    sep = "=" * 60
    print(f"\n{sep}")
    print("Starting RL Algorithm Comparison")
    print(sep)
    print(f"Environment : {args.env}")
    print(f"Algorithms  : {', '.join(args.algos)}")
    print(f"Seeds       : {args.seeds}")
    print(f"Timesteps   : {args.timesteps:,}")
    print(f"Total runs  : {len(args.algos) * len(args.seeds)}")
    print(f"{sep}\n")

    all_results: List[Dict] = []

    for algo in args.algos:
        for seed in args.seeds:
            print(f"\n{sep}")
            print(f"Training {algo} on {args.env}  (seed={seed})")
            print(sep)

            res = train_and_evaluate(
                algo,
                args.env,
                args.timesteps,
                args.eval_episodes,
                seed,
                out_dir / algo / f"seed_{seed}",
            )
            all_results.append(res)

            print(f"\n{algo} seed={seed} results:")
            print(f"  Mean reward   : {res['mean_reward']:.1f} ± {res['std_reward']:.1f}")
            print(f"  Median reward : {res['median_reward']:.1f}")
            print(f"  Min / Max     : {res['min_reward']:.1f} / {res['max_reward']:.1f}")
            print(f"  Train time    : {res['train_time'] / 60:.1f} min")
            print(f"  Peak RAM      : {res['peak_memory_mb']:.1f} MB")

    summary_df, detailed_df = aggregate_results(all_results)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    detailed_df.to_csv(out_dir / "detailed_results.csv", index=False)

    statistical_tests(detailed_df, out_dir)
    plot_results(summary_df, detailed_df, all_results, out_dir, args.timesteps)

    print(f"\n{sep}")
    print("FINAL RESULTS SUMMARY")
    print(sep)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {out_dir}")
    print(sep)


if __name__ == "__main__":
    main()
