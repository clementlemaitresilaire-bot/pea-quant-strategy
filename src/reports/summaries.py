from __future__ import annotations

from collections import Counter

import pandas as pd

from src.engines.allocation_engine import build_full_target_allocation
from src.engines.order_engine import build_order_proposals


def build_run_summary(
    allocation_result=None,
    order_proposals_df: pd.DataFrame | None = None,
) -> str:
    """
    Build a human-readable summary of the current strategy run.
    """
    if allocation_result is None:
        allocation_result = build_full_target_allocation()

    if order_proposals_df is None:
        order_proposals_df = build_order_proposals(
            combined_target_df=allocation_result.combined_target_weights_table
        )

    core_df = allocation_result.core_target_weights_table
    opp_df = allocation_result.opp_target_weights_table
    combined_df = allocation_result.combined_target_weights_table

    executable_orders = order_proposals_df.loc[order_proposals_df["execute"] == True].copy()
    rejected_orders = order_proposals_df.loc[order_proposals_df["execute"] == False].copy()

    reject_reason_counts = Counter(rejected_orders["decision_reason"].tolist())
    execute_reason_counts = Counter(executable_orders["decision_reason"].tolist())

    lines: list[str] = []
    lines.append("=== Strategy Run Summary ===")
    lines.append("")
    lines.append(f"Market regime              : {allocation_result.regime}")
    lines.append(f"Target core deployment     : {allocation_result.target_core_deployment:.2%}")
    lines.append(f"Target core weight total   : {allocation_result.target_core_weight_total:.2%}")
    lines.append(f"Target opp weight total    : {allocation_result.target_opp_weight_total:.2%}")
    lines.append(f"Target invested total      : {allocation_result.target_total_invested_weight:.2%}")
    lines.append(f"Implied cash weight        : {allocation_result.implied_cash_weight:.2%}")
    lines.append("")
    lines.append(f"Selected core names        : {len(core_df)}")
    lines.append(f"Selected opp names         : {len(opp_df)}")
    lines.append(f"Combined target positions  : {len(combined_df)}")
    lines.append("")
    lines.append(f"Executable orders          : {len(executable_orders)}")
    lines.append(f"Rejected / no-action rows  : {len(rejected_orders)}")

    if not executable_orders.empty:
        lines.append("")
        lines.append("Executable order reasons:")
        for reason, count in sorted(execute_reason_counts.items()):
            lines.append(f"  - {reason}: {count}")

    if reject_reason_counts:
        lines.append("")
        lines.append("Rejected / no-action reasons:")
        for reason, count in sorted(reject_reason_counts.items()):
            lines.append(f"  - {reason}: {count}")

    if not executable_orders.empty:
        lines.append("")
        lines.append("Executable orders preview:")
        preview_cols = ["ticker", "bucket", "action", "order_value", "total_cost_est", "decision_reason"]
        preview_df = executable_orders[preview_cols].copy()

        for _, row in preview_df.iterrows():
            lines.append(
                f"  - {row['ticker']} | {row['bucket']} | {row['action']} | "
                f"order={row['order_value']:.2f} EUR | cost={row['total_cost_est']:.2f} EUR | "
                f"{row['decision_reason']}"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    summary_text = build_run_summary()
    print(summary_text)