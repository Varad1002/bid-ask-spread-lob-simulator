"""
visualization.py -- Generate all required plots from simulation data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict

COLORS = {
    "balanced": "#2196F3",
    "buyer_heavy": "#4CAF50",
    "seller_heavy": "#F44336",
    "high_cancel": "#FF9800",
    "low_liquidity": "#9C27B0",
}


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#555555",
        "legend.fontsize": 10,
        "grid.color": "#333366",
        "grid.alpha": 0.5,
        "lines.linewidth": 1.5,
        "font.family": "sans-serif",
        "savefig.bbox": "tight",
        "savefig.facecolor": "#1a1a2e",
    })


def ensure_output_dir(d: str):
    os.makedirs(d, exist_ok=True)


# -- Plot 1: Spread vs Time --

def plot_spread_vs_time(results: Dict[str, pd.DataFrame], output_dir: str):
    setup_style()
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3.5 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, results.items()):
        valid = df.dropna(subset=["spread"])
        color = COLORS.get(name, "#2196F3")

        ax.plot(valid["step"], valid["spread"], alpha=0.25, color=color, linewidth=0.5)

        window = min(500, len(valid) // 10)
        if window > 1:
            rolling = valid["spread"].rolling(window=window).mean()
            ax.plot(valid["step"], rolling, color=color, linewidth=2,
                    label=f"Rolling avg ({window})")

        ax.set_ylabel("Spread")
        ax.set_title(df["experiment_label"].iloc[0], fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle("Bid-Ask Spread Over Time", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_vs_time.png"))
    plt.close()
    print("  [OK] spread_vs_time.png")


# -- Plot 2: Spread vs Liquidity --

def plot_spread_vs_liquidity(liquidity_df: pd.DataFrame, output_dir: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        liquidity_df["initial_orders"],
        liquidity_df["mean_spread"],
        yerr=liquidity_df["std_spread"],
        marker="o", markersize=8, color="#00BCD4", ecolor="#80DEEA",
        capsize=5, linewidth=2, markerfacecolor="#E0F7FA", markeredgecolor="#00BCD4",
    )
    ax.set_xlabel("Initial Number of Orders (Liquidity)")
    ax.set_ylabel("Mean Spread")
    ax.set_title("Bid-Ask Spread vs Liquidity", fontsize=14, fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_vs_liquidity.png"))
    plt.close()
    print("  [OK] spread_vs_liquidity.png")


# -- Plot 3: Spread vs Order Imbalance --

def plot_spread_vs_imbalance(imbalance_df: pd.DataFrame, output_dir: str):
    setup_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.errorbar(
        imbalance_df["buy_ratio"], imbalance_df["mean_spread"],
        yerr=imbalance_df["std_spread"],
        marker="s", markersize=8, color="#FF5722", ecolor="#FFAB91",
        capsize=5, linewidth=2, markerfacecolor="#FBE9E7", markeredgecolor="#FF5722",
        label="Mean Spread",
    )
    ax1.set_xlabel("Buy Ratio (fraction of orders that are buys)")
    ax1.set_ylabel("Mean Spread", color="#FF5722")
    ax1.tick_params(axis="y", labelcolor="#FF5722")

    ax2 = ax1.twinx()
    ax2.bar(
        imbalance_df["buy_ratio"], imbalance_df["mean_imbalance"],
        width=0.06, alpha=0.4, color="#7C4DFF", label="Mean Imbalance",
    )
    ax2.set_ylabel("Mean Order Imbalance (Buy - Sell)", color="#7C4DFF")
    ax2.tick_params(axis="y", labelcolor="#7C4DFF")

    ax1.set_title("Bid-Ask Spread vs Order Imbalance", fontsize=14, fontweight="bold")
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_vs_imbalance.png"))
    plt.close()
    print("  [OK] spread_vs_imbalance.png")


# -- Plot 4: Spread vs Cancellation Rate --

def plot_spread_vs_cancellation(cancel_df: pd.DataFrame, output_dir: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        cancel_df["cancellation_rate"], cancel_df["mean_spread"],
        yerr=cancel_df["std_spread"],
        marker="D", markersize=8, color="#FFEB3B", ecolor="#FFF9C4",
        capsize=5, linewidth=2, markerfacecolor="#FFFDE7", markeredgecolor="#F9A825",
    )
    ax.set_xlabel("Cancellations Per Step")
    ax.set_ylabel("Mean Spread")
    ax.set_title("Bid-Ask Spread vs Cancellation Rate", fontsize=14, fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_vs_cancellation.png"))
    plt.close()
    print("  [OK] spread_vs_cancellation.png")


# -- Plot 5: Spread Distribution --

def plot_spread_distribution(results: Dict[str, pd.DataFrame], output_dir: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, df in results.items():
        valid = df.dropna(subset=["spread"])
        color = COLORS.get(name, "#2196F3")
        label = df["experiment_label"].iloc[0]
        ax.hist(valid["spread"], bins=80, alpha=0.45, color=color,
                label=label, edgecolor=color, linewidth=0.5)

    ax.set_xlabel("Spread")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Bid-Ask Spread", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_distribution.png"))
    plt.close()
    print("  [OK] spread_distribution.png")


# -- Plot 6: Mid-Price Over Time --

def plot_mid_price(results: Dict[str, pd.DataFrame], output_dir: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, df in results.items():
        valid = df.dropna(subset=["mid_price"])
        color = COLORS.get(name, "#2196F3")
        label = df["experiment_label"].iloc[0]
        window = min(500, len(valid) // 10)
        if window > 1:
            rolling = valid["mid_price"].rolling(window=window).mean()
            ax.plot(valid["step"], rolling, color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mid-Price")
    ax.set_title("Mid-Price Evolution Over Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mid_price_evolution.png"))
    plt.close()
    print("  [OK] mid_price_evolution.png")


# -- Generate All --

def generate_all_plots(results, liquidity_df, imbalance_df, cancel_df, output_dir="output"):
    ensure_output_dir(output_dir)
    print("\n[PLOTS] Generating plots...")
    plot_spread_vs_time(results, output_dir)
    plot_spread_vs_liquidity(liquidity_df, output_dir)
    plot_spread_vs_imbalance(imbalance_df, output_dir)
    plot_spread_vs_cancellation(cancel_df, output_dir)
    plot_spread_distribution(results, output_dir)
    plot_mid_price(results, output_dir)
    print(f"[PLOTS] All plots saved to '{output_dir}/'")
