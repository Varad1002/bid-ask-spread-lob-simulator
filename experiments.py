"""
experiments.py -- Experiments with stochastic LOB under different conditions.
"""

from typing import Dict
import pandas as pd
from simulator import Simulator, SimulationConfig


def get_experiment_configs() -> Dict[str, SimulationConfig]:
    """Five main experiments with 100K initial orders."""
    configs = {}

    # 1. Balanced market
    configs["balanced"] = SimulationConfig(
        num_steps=20_000,
        initial_orders=100_000,
        initial_sigma=2.0,
        order_sigma=0.30,
        mid_price_volatility=0.02,
        buy_ratio=0.5,
        limit_orders_per_step=3,
        market_order_prob=0.5,
        market_qty_min=10,
        market_qty_max=50,
        cancels_per_step=5,
        stale_distance=1.0,
        seed=42,
        label="Balanced Market",
    )

    # 2. Buyer-heavy (70% buy orders)
    configs["buyer_heavy"] = SimulationConfig(
        num_steps=20_000,
        initial_orders=100_000,
        initial_sigma=2.0,
        order_sigma=0.30,
        mid_price_volatility=0.02,
        buy_ratio=0.7,
        limit_orders_per_step=3,
        market_order_prob=0.5,
        market_qty_min=10,
        market_qty_max=50,
        cancels_per_step=5,
        stale_distance=1.0,
        seed=42,
        label="Buyer-Heavy (70%)",
    )

    # 3. Seller-heavy (30% buy = 70% sell)
    configs["seller_heavy"] = SimulationConfig(
        num_steps=20_000,
        initial_orders=100_000,
        initial_sigma=2.0,
        order_sigma=0.30,
        mid_price_volatility=0.02,
        buy_ratio=0.3,
        limit_orders_per_step=3,
        market_order_prob=0.5,
        market_qty_min=10,
        market_qty_max=50,
        cancels_per_step=5,
        stale_distance=1.0,
        seed=42,
        label="Seller-Heavy (70%)",
    )

    # 4. High cancellation rate (15 cancels/step vs 5)
    configs["high_cancel"] = SimulationConfig(
        num_steps=20_000,
        initial_orders=100_000,
        initial_sigma=2.0,
        order_sigma=0.30,
        mid_price_volatility=0.02,
        buy_ratio=0.5,
        limit_orders_per_step=3,
        market_order_prob=0.5,
        market_qty_min=10,
        market_qty_max=50,
        cancels_per_step=15,
        stale_distance=1.0,
        seed=42,
        label="High Cancellation",
    )

    # 5. Low liquidity (10K initial, fewer limit orders per step)
    configs["low_liquidity"] = SimulationConfig(
        num_steps=20_000,
        initial_orders=10_000,
        initial_sigma=2.0,
        order_sigma=0.30,
        mid_price_volatility=0.02,
        buy_ratio=0.5,
        limit_orders_per_step=1,
        market_order_prob=0.5,
        market_qty_min=10,
        market_qty_max=50,
        cancels_per_step=5,
        stale_distance=1.0,
        seed=42,
        label="Low Liquidity",
    )

    return configs


def run_all_experiments() -> Dict[str, pd.DataFrame]:
    configs = get_experiment_configs()
    results = {}
    for name, config in configs.items():
        sim = Simulator(config)
        df = sim.run()
        df["experiment"] = name
        df["experiment_label"] = config.label
        results[name] = df
    return results


# -- Parameter sweeps (lighter configs for speed) --

def _sweep_config(**overrides) -> SimulationConfig:
    """Base config for sweep simulations (smaller scale for speed)."""
    base = dict(
        num_steps=5_000,
        initial_orders=5_000,
        initial_sigma=2.0,
        order_sigma=0.30,
        mid_price_volatility=0.02,
        buy_ratio=0.5,
        limit_orders_per_step=3,
        market_order_prob=0.5,
        market_qty_min=10,
        market_qty_max=50,
        cancels_per_step=5,
        stale_distance=1.0,
        seed=42,
    )
    base.update(overrides)
    return SimulationConfig(**base)


def run_liquidity_sweep() -> pd.DataFrame:
    """Vary initial_orders to see spread vs liquidity."""
    records = []
    for liq in [500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000]:
        # Scale limit_orders_per_step proportionally
        lo_rate = max(1, liq // 5000)
        cfg = _sweep_config(
            initial_orders=liq,
            limit_orders_per_step=min(lo_rate, 5),
            label=f"Liquidity={liq}",
        )
        sim = Simulator(cfg)
        df = sim.run()
        valid = df.dropna(subset=["spread"])
        if len(valid) > 0:
            records.append({
                "initial_orders": liq,
                "mean_spread": valid["spread"].mean(),
                "std_spread": valid["spread"].std(),
                "mean_total_orders": valid["total_orders"].mean(),
            })
    return pd.DataFrame(records)


def run_imbalance_sweep() -> pd.DataFrame:
    """Vary buy_ratio to see spread vs order imbalance."""
    records = []
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cfg = _sweep_config(
            buy_ratio=ratio,
            label=f"BuyRatio={ratio}",
        )
        sim = Simulator(cfg)
        df = sim.run()
        valid = df.dropna(subset=["spread"])
        if len(valid) > 0:
            records.append({
                "buy_ratio": ratio,
                "mean_spread": valid["spread"].mean(),
                "std_spread": valid["spread"].std(),
                "mean_imbalance": (valid["num_buy_orders"] - valid["num_sell_orders"]).mean(),
            })
    return pd.DataFrame(records)


def run_cancellation_sweep() -> pd.DataFrame:
    """Vary cancels_per_step to see spread vs cancellation intensity."""
    records = []
    for cancels in [0, 1, 2, 5, 8, 10, 15, 20, 30]:
        cfg = _sweep_config(
            cancels_per_step=cancels,
            label=f"Cancels={cancels}",
        )
        sim = Simulator(cfg)
        df = sim.run()
        valid = df.dropna(subset=["spread"])
        if len(valid) > 0:
            records.append({
                "cancellation_rate": cancels,
                "mean_spread": valid["spread"].mean(),
                "std_spread": valid["spread"].std(),
            })
    return pd.DataFrame(records)
