"""
main.py -- Entry point for the Limit Order Book simulation and analysis.
"""

import time
import os

from experiments import (
    run_all_experiments,
    run_liquidity_sweep,
    run_imbalance_sweep,
    run_cancellation_sweep,
)
from visualization import generate_all_plots
from analysis import compute_statistics, print_statistics, export_all_data


def main():
    print("=" * 70)
    print("  LIMIT ORDER BOOK SIMULATOR (Stochastic Version)")
    print("  Dynamic Bid-Ask Spread Under Different Market Conditions")
    print("=" * 70)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    # -- Phase 1: Main experiments (100K initial orders each) --
    print("\n[1/4] Running main experiments...")
    results = run_all_experiments()

    # -- Phase 2: Parameter sweeps (lighter configs) --
    print("\n[2/4] Running parameter sweeps...")
    print("  -> Liquidity sweep...")
    liquidity_df = run_liquidity_sweep()
    print("  -> Imbalance sweep...")
    imbalance_df = run_imbalance_sweep()
    print("  -> Cancellation rate sweep...")
    cancel_df = run_cancellation_sweep()

    # -- Phase 3: Visualizations --
    print("\n[3/4] Generating visualizations...")
    generate_all_plots(results, liquidity_df, imbalance_df, cancel_df, output_dir)

    # -- Phase 4: Statistics --
    print("\n[4/4] Computing statistics...")
    stats_df = compute_statistics(results)
    print_statistics(stats_df)
    export_all_data(results, stats_df, output_dir)

    elapsed = time.time() - start
    print(f"\n[DONE] All done in {elapsed:.1f} seconds.")
    print(f"   Outputs saved to: {os.path.abspath(output_dir)}/")
    print()
    print("Generated files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"   - {f} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
