"""
analysis.py -- Statistical analysis of simulation results.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict


def compute_statistics(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for name, df in results.items():
        valid = df.dropna(subset=["spread"])
        label = df["experiment_label"].iloc[0] if len(df) > 0 else name
        if len(valid) == 0:
            continue

        spread = valid["spread"]
        imbalance = valid["num_buy_orders"] - valid["num_sell_orders"]
        liquidity = valid["total_orders"]

        corr_liq = spread.corr(liquidity) if spread.std() > 0 and len(spread) > 2 else np.nan
        corr_imb = spread.corr(imbalance.abs()) if spread.std() > 0 and len(spread) > 2 else np.nan

        records.append({
            "Experiment": label,
            "Mean Spread": round(spread.mean(), 4),
            "Variance": round(spread.var(), 6),
            "Std Dev": round(spread.std(), 4),
            "Median": round(spread.median(), 4),
            "Min Spread": round(spread.min(), 4),
            "Max Spread": round(spread.max(), 4),
            "Mean Mid-Price": round(valid["mid_price"].mean(), 4) if "mid_price" in valid else np.nan,
            "Total Trades": int(df["num_trades"].sum()),
            "Total Volume": int(df["volume_traded"].sum()),
            "Corr(Spread, Liquidity)": round(corr_liq, 4) if not np.isnan(corr_liq) else "N/A",
            "Corr(Spread, |Imbalance|)": round(corr_imb, 4) if not np.isnan(corr_imb) else "N/A",
        })

    return pd.DataFrame(records)


def print_statistics(stats_df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("  STATISTICAL ANALYSIS -- BID-ASK SPREAD ACROSS EXPERIMENTS")
    print("=" * 100)
    print()

    for _, row in stats_df.iterrows():
        name = str(row['Experiment'])
        print(f"  +-- {name} {'-' * (60 - len(name))}+")
        print(f"  |  Mean Spread          : {row['Mean Spread']:>10}")
        print(f"  |  Variance             : {row['Variance']:>12}")
        print(f"  |  Std Deviation        : {row['Std Dev']:>10}")
        print(f"  |  Median Spread        : {row['Median']:>10}")
        print(f"  |  Min / Max Spread     : {row['Min Spread']:>8} / {row['Max Spread']:<8}")
        print(f"  |  Mean Mid-Price       : {row['Mean Mid-Price']:>10}")
        print(f"  |  Total Trades         : {row['Total Trades']:>10}")
        print(f"  |  Total Volume         : {row['Total Volume']:>10}")
        print(f"  |  Corr(Spread, Liq.)   : {str(row['Corr(Spread, Liquidity)']):>10}")
        print(f"  |  Corr(Spread, |Imb.|) : {str(row['Corr(Spread, |Imbalance|)']):>10}")
        print(f"  +{'-' * 64}+")
        print()


def export_all_data(results, stats_df, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    for name, df in results.items():
        df.to_csv(os.path.join(output_dir, f"data_{name}.csv"), index=False)

    combined = pd.concat(results.values(), ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "data_all_experiments.csv"), index=False)

    stats_df.to_csv(os.path.join(output_dir, "statistics_summary.csv"), index=False)

    print(f"[DATA] Data exported to '{output_dir}/'")
