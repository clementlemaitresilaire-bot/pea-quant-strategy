from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.engines.allocation_engine import build_full_target_allocation
from src.engines.core_engine import build_core_signal_table
from src.engines.opp_engine import build_opp_signal_table
from src.engines.order_engine import build_order_proposals
from src.settings import EXPORT_DATA_DIR, ensure_project_directories


def export_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """
    Export a DataFrame to data/exports/<filename>.
    """
    ensure_project_directories()

    output_path = EXPORT_DATA_DIR / filename
    df.to_csv(output_path, index=False)

    return output_path


def export_all_strategy_outputs() -> dict[str, Path]:
    """
    Run the main strategy engines and export all key outputs.
    """
    core_signal_df = build_core_signal_table()
    opp_signal_df = build_opp_signal_table()
    allocation_result = build_full_target_allocation(
        core_signal_df=core_signal_df,
        opp_signal_df=opp_signal_df,
    )
    orders_df = build_order_proposals(
        combined_target_df=allocation_result.combined_target_weights_table
    )

    exported_files = {
        "core_signals": export_dataframe(core_signal_df, "core_signals.csv"),
        "opp_signals": export_dataframe(opp_signal_df, "opp_signals.csv"),
        "core_target_weights": export_dataframe(
            allocation_result.core_target_weights_table,
            "core_target_weights.csv",
        ),
        "opp_target_weights": export_dataframe(
            allocation_result.opp_target_weights_table,
            "opp_target_weights.csv",
        ),
        "combined_target_weights": export_dataframe(
            allocation_result.combined_target_weights_table,
            "combined_target_weights.csv",
        ),
        "order_proposals": export_dataframe(
            orders_df,
            "order_proposals.csv",
        ),
    }

    return exported_files


if __name__ == "__main__":
    exported = export_all_strategy_outputs()

    print("\n=== Exported Files ===")
    for name, path in exported.items():
        print(f"{name}: {path}")