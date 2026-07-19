"""Build executive tables and charts for the consolidated Model_D reassessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MISSING_COLOR = "#d1d5db"
LOW_DATA_COLOR = "#f59e0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final Model_D reassessment tables and charts.")
    parser.add_argument(
        "--input-dir",
        default="src/predictions/Model_D/final_reassessment",
        help="Directory containing consolidated reassessment CSV artifacts.",
    )
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _build_horizon_months(metadata: dict[str, object], summary: pd.DataFrame) -> list[str]:
    months = metadata.get("months_requested")
    if isinstance(months, list) and months:
        return [str(month) for month in months]
    return sorted(summary["source_month"].astype(str).unique().tolist())


def _prepare_plot_pivot(summary: pd.DataFrame, value_column: str, horizon_months: list[str]) -> pd.DataFrame:
    pivot = summary.pivot(index="source_month", columns="mode", values=value_column)
    pivot.index = pivot.index.astype(str)
    return pivot.reindex(horizon_months)


def _annotate_caveats(ax: plt.Axes, horizon_months: list[str], salvedades: dict[str, object]) -> None:
    missing_months = {str(month) for month in salvedades.get("missing_months_in_requested_horizon", [])}
    low_data_months = {
        str(item.get("source_month"))
        for item in salvedades.get("low_data_months", [])
        if isinstance(item, dict) and item.get("source_month") is not None
    }
    xticks = ax.get_xticks()
    for idx, month in enumerate(horizon_months):
        if idx >= len(xticks):
            continue
        center = xticks[idx]
        if month in missing_months:
            ax.axvspan(center - 0.5, center + 0.5, color=MISSING_COLOR, alpha=0.35, zorder=0)
        elif month in low_data_months:
            ax.axvspan(center - 0.5, center + 0.5, color=LOW_DATA_COLOR, alpha=0.18, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    extra_handles = []
    extra_labels = []
    if missing_months:
        extra_handles.append(plt.Rectangle((0, 0), 1, 1, color=MISSING_COLOR, alpha=0.35))
        extra_labels.append("Missing month")
    if low_data_months:
        extra_handles.append(plt.Rectangle((0, 0), 1, 1, color=LOW_DATA_COLOR, alpha=0.18))
        extra_labels.append("Low-data month")
    if extra_handles:
        ax.legend(handles + extra_handles, labels + extra_labels, title="Legend")


def _finalize_axis(ax: plt.Axes, horizon_months: list[str], title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Source month")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(horizon_months)))
    ax.set_xticklabels(horizon_months, rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def _plot_runtime(summary: pd.DataFrame, output_path: Path, horizon_months: list[str], salvedades: dict[str, object]) -> None:
    pivot = _prepare_plot_pivot(summary, "elapsed_seconds", horizon_months)
    ax = pivot.plot(kind="bar", figsize=(18, 6), color=["#1f77b4", "#ff7f0e"])
    _finalize_axis(ax, horizon_months, "Model_D runtime by month and detection mode", "Elapsed seconds")
    _annotate_caveats(ax, horizon_months, salvedades)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_episodes(summary: pd.DataFrame, output_path: Path, horizon_months: list[str], salvedades: dict[str, object]) -> None:
    pivot = _prepare_plot_pivot(summary, "episode_count", horizon_months)
    ax = pivot.plot(kind="bar", figsize=(18, 6), color=["#2a9d8f", "#e76f51"])
    _finalize_axis(ax, horizon_months, "Detected episodes by month and detection mode", "Episode count")
    _annotate_caveats(ax, horizon_months, salvedades)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_recall(summary: pd.DataFrame, output_path: Path, horizon_months: list[str], salvedades: dict[str, object]) -> None:
    pivot = _prepare_plot_pivot(summary, "episode_recall", horizon_months)
    ax = pivot.plot(kind="line", marker="o", figsize=(18, 6), color=["#264653", "#f4a261"])
    _finalize_axis(ax, horizon_months, "Episode recall by month and detection mode", "Episode recall")
    ax.set_ylim(0.0, max(0.25, float(summary["episode_recall"].max()) * 1.1))
    ax.grid(alpha=0.25)
    _annotate_caveats(ax, horizon_months, salvedades)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_matches(summary: pd.DataFrame, output_path: Path, horizon_months: list[str], salvedades: dict[str, object]) -> None:
    pivot = _prepare_plot_pivot(summary, "match_count", horizon_months)
    ax = pivot.plot(kind="bar", figsize=(18, 6), color=["#577590", "#f94144"])
    _finalize_axis(ax, horizon_months, "Registry matches by month and detection mode", "Match count")
    _annotate_caveats(ax, horizon_months, salvedades)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _build_mode_table(summary: pd.DataFrame) -> pd.DataFrame:
    mode_table = (
        summary.groupby("mode", dropna=False)
        .agg(
            months_processed=("source_month", "count"),
            total_episodes=("episode_count", "sum"),
            total_windows=("window_count", "sum"),
            total_matches=("match_count", "sum"),
            mean_runtime_seconds=("elapsed_seconds", "mean"),
            median_runtime_seconds=("elapsed_seconds", "median"),
            mean_episode_recall=("episode_recall", "mean"),
            mean_temporal_iou=("mean_temporal_iou", "mean"),
            mean_family_recall=("family_recall", "mean"),
        )
        .reset_index()
    )
    return mode_table.sort_values("mode").reset_index(drop=True)


def _build_month_table(summary: pd.DataFrame, horizon_months: list[str], salvedades: dict[str, object]) -> pd.DataFrame:
    month_table = (
        summary.pivot_table(
            index="source_month",
            columns="mode",
            values=["episode_count", "match_count", "elapsed_seconds", "episode_recall"],
        )
    )
    month_table.columns = [f"{metric}_{mode}" for metric, mode in month_table.columns]
    month_table = month_table.reindex(horizon_months)
    month_table.index.name = "source_month"
    month_table = month_table.reset_index()
    missing_months = {str(month) for month in salvedades.get("missing_months_in_requested_horizon", [])}
    low_data_map = {
        str(item.get("source_month")): item
        for item in salvedades.get("low_data_months", [])
        if isinstance(item, dict) and item.get("source_month") is not None
    }
    statuses = []
    for month in month_table["source_month"].astype(str):
        if month in missing_months:
            statuses.append("missing")
        elif month in low_data_map:
            statuses.append("low_data")
        else:
            statuses.append("processed")
    month_table.insert(1, "horizon_status", statuses)
    return month_table


def _build_highlights(summary: pd.DataFrame) -> dict[str, object]:
    semantic = summary[summary["mode"] == "semantic"].copy()
    indicators = summary[summary["mode"] == "indicators"].copy()
    best_recall_row = summary.sort_values(["episode_recall", "match_count"], ascending=[False, False]).iloc[0]
    slowest_row = summary.sort_values("elapsed_seconds", ascending=False).iloc[0]
    return {
        "months_processed": int(summary["source_month"].nunique()),
        "best_recall_month": str(best_recall_row["source_month"]),
        "best_recall_mode": str(best_recall_row["mode"]),
        "best_recall_value": float(best_recall_row["episode_recall"]),
        "slowest_month": str(slowest_row["source_month"]),
        "slowest_mode": str(slowest_row["mode"]),
        "slowest_runtime_seconds": float(slowest_row["elapsed_seconds"]),
        "semantic_mean_runtime_seconds": float(semantic["elapsed_seconds"].mean()),
        "indicator_mean_runtime_seconds": float(indicators["elapsed_seconds"].mean()),
        "semantic_total_episodes": int(semantic["episode_count"].sum()),
        "indicator_total_episodes": int(indicators["episode_count"].sum()),
    }


def _format_salvedades_html(salvedades: dict[str, object]) -> str:
    missing_months = [str(month) for month in salvedades.get("missing_months_in_requested_horizon", [])]
    low_data_months = salvedades.get("low_data_months", [])
    note = str(salvedades.get("precision_interpretation_note", ""))
    items = []
    if missing_months:
        items.append(f"<li>Missing months in horizon: {', '.join(missing_months)}</li>")
    if low_data_months:
        details = []
        for item in low_data_months:
            if not isinstance(item, dict):
                continue
            details.append(
                f"{item.get('source_month')}: timeline_rows={item.get('timeline_rows')}, sequence_rows={item.get('sequence_rows')}, assignment_rows={item.get('assignment_rows')}"
            )
        items.append(f"<li>Low-data months: {'; '.join(details)}</li>")
    if note:
        items.append(f"<li>Metric note: {note}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else "<p>No caveats reported.</p>"


def _build_html_report(
    summary: pd.DataFrame,
    mode_table: pd.DataFrame,
    month_table: pd.DataFrame,
    highlights: dict[str, object],
    output_dir: Path,
    salvedades: dict[str, object],
) -> str:
    runtime_png = "runtime_by_month.png"
    episodes_png = "episodes_by_month.png"
    recall_png = "recall_by_month.png"
    matches_png = "matches_by_month.png"
    salvedades_html = _format_salvedades_html(salvedades)
    return f"""<html>
<head>
  <meta charset="utf-8">
  <title>Model_D Final Reassessment Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 24px; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f59e0b; padding: 12px; }}
    .caveat {{ background: #f8fafc; border-left: 4px solid #64748b; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Model_D Final Reassessment</h1>
  <p>Generated from consolidated multi-month reassessment artifacts in <code>{output_dir}</code>.</p>
  <h2>Highlights</h2>
  <ul>
    <li>Months processed: {highlights['months_processed']}</li>
    <li>Best recall: {highlights['best_recall_value']:.4f} in {highlights['best_recall_month']} using {highlights['best_recall_mode']}</li>
    <li>Mean runtime: semantic {highlights['semantic_mean_runtime_seconds']:.3f}s, indicators {highlights['indicator_mean_runtime_seconds']:.3f}s</li>
    <li>Total episodes: semantic {highlights['semantic_total_episodes']}, indicators {highlights['indicator_total_episodes']}</li>
    <li>Slowest run: {highlights['slowest_month']} {highlights['slowest_mode']} at {highlights['slowest_runtime_seconds']:.3f}s</li>
  </ul>
  <div class="note">
    The current evaluator yields very high precision-like values because registry matches are counted at episode level with repeated alignments. Treat recall and temporal overlap as more reliable comparative signals than raw precision for executive interpretation.
  </div>
  <h2>Horizon Caveats</h2>
  <div class="caveat">{salvedades_html}</div>
  <h2>Mode Summary</h2>
  {mode_table.to_html(index=False)}
  <h2>Month Summary</h2>
  {month_table.to_html(index=False)}
  <h2>Charts</h2>
  <p>Gray bands mark missing months in the requested horizon. Amber bands mark low-data months.</p>
  <h3>Runtime by Month</h3>
  <img src="{runtime_png}" alt="Runtime by month">
  <h3>Detected Episodes by Month</h3>
  <img src="{episodes_png}" alt="Episodes by month">
  <h3>Episode Recall by Month</h3>
  <img src="{recall_png}" alt="Recall by month">
  <h3>Registry Matches by Month</h3>
  <img src="{matches_png}" alt="Matches by month">
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    summary = _load_csv(input_dir / "model_d_reassessment_summary.csv")
    metadata = _load_json(input_dir / "model_d_reassessment_metadata.json")
    salvedades = _load_json(input_dir / "model_d_reassessment_salvedades.json")
    horizon_months = _build_horizon_months(metadata, summary)

    mode_table = _build_mode_table(summary)
    month_table = _build_month_table(summary, horizon_months, salvedades)
    highlights = _build_highlights(summary)

    _save_table(mode_table, input_dir / "model_d_final_mode_summary.csv")
    _save_table(month_table, input_dir / "model_d_final_month_summary.csv")
    (input_dir / "model_d_final_highlights.json").write_text(json.dumps(highlights, indent=2), encoding="utf-8")

    _plot_runtime(summary, input_dir / "runtime_by_month.png", horizon_months, salvedades)
    _plot_episodes(summary, input_dir / "episodes_by_month.png", horizon_months, salvedades)
    _plot_recall(summary, input_dir / "recall_by_month.png", horizon_months, salvedades)
    _plot_matches(summary, input_dir / "matches_by_month.png", horizon_months, salvedades)

    report_html = _build_html_report(summary, mode_table, month_table, highlights, input_dir, salvedades)
    (input_dir / "model_d_final_report.html").write_text(report_html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
