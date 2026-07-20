from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
MAINTENANCE_DOC = Path("/tmp/Tabla_Incidencias.docx")
OT_DOC = Path("/tmp/OT - INCIDENCIAS LAVA RUEDAS.docx")
DEFAULT_DATASET = "probe"
DATASET_ROOTS = {
    "probe": Path("src/predictions/episode_eval_probe"),
    "full": Path("src/predictions/episode_eval_full"),
}
FOCUS_START = pd.Timestamp("2024-02-01 00:00:00")
FOCUS_END = pd.Timestamp("2026-04-30 23:59:59")


def _extract_docx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    table = root.findall(".//w:tbl", NS)[0]
    rows: list[list[str]] = []
    for tr in table.findall("./w:tr", NS):
        cells: list[str] = []
        for tc in tr.findall("./w:tc", NS):
            texts = [node.text or "" for node in tc.findall(".//w:t", NS)]
            cells.append("".join(texts).strip())
        rows.append(cells)
    return rows


def _parse_concat_range(text: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    matches = re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", text or "")
    if len(matches) < 2:
        return None, None
    return pd.Timestamp(matches[0].replace("T", " ")), pd.Timestamp(matches[1].replace("T", " "))


def _parse_dates(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(item) for item in re.findall(r"\d{4}-\d{2}-\d{2}", text or "")]


def load_maintenance_periods() -> pd.DataFrame:
    rows = _extract_docx_rows(MAINTENANCE_DOC)[1:]
    periods: list[dict[str, object]] = []
    for row in rows:
        start, end = _parse_concat_range(row[0])
        if start is None or end is None:
            continue
        if end < FOCUS_START or start > FOCUS_END:
            continue
        periods.append(
            {
                "start": max(start, FOCUS_START),
                "end": min(end, FOCUS_END),
                "label": row[2] or "SIN COMENTARIOS",
                "source": "Tabla incidencias",
            }
        )
    return pd.DataFrame(periods).sort_values("start").reset_index(drop=True)


def load_ot_points() -> pd.DataFrame:
    rows = _extract_docx_rows(OT_DOC)[2:]
    records: list[dict[str, object]] = []
    current_window = ""
    for row in rows:
        if row and row[0].strip():
            current_window = row[0].strip()
        part_date = row[1].strip() if len(row) > 1 else ""
        label = row[2].strip() if len(row) > 2 else ""
        for date in _parse_dates(part_date):
            if not (FOCUS_START <= date <= FOCUS_END):
                continue
            records.append(
                {
                    "date": date,
                    "label": label or "(sin etiqueta)",
                    "window": current_window,
                    "is_unavailability": _is_unavailability_label(label),
                }
            )
    frame = pd.DataFrame(records).drop_duplicates().sort_values("date").reset_index(drop=True)
    return frame


def _is_unavailability_label(label: str) -> bool:
    text = (label or "").lower()
    keywords = (
        "no funciona",
        "no operativo",
        "bomba no operativa",
        "bomba de foso no funciona",
        "fallo funcionamiento",
        "desbordando",
        "rebosando",
    )
    return any(keyword in text for keyword in keywords)


def load_episodes(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "detected_episodes.csv", parse_dates=["event_start", "event_end"])
    frame = frame[(frame["event_end"] >= FOCUS_START) & (frame["event_start"] <= FOCUS_END)].copy()
    frame["event_start"] = frame["event_start"].clip(lower=FOCUS_START)
    frame["event_end"] = frame["event_end"].clip(upper=FOCUS_END)
    frame["duration_hours"] = (frame["event_end"] - frame["event_start"]).dt.total_seconds() / 3600.0
    return frame.sort_values("event_start").reset_index(drop=True)


def load_windows(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "window_scores.csv", parse_dates=["start_time", "end_time"])
    frame = frame[(frame["end_time"] >= FOCUS_START) & (frame["start_time"] <= FOCUS_END)].copy()
    return frame.sort_values("start_time").reset_index(drop=True)


def plot_episode_timeline(periods: pd.DataFrame, ot_points: pd.DataFrame, baseline: pd.DataFrame, localized: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True, constrained_layout=True)
    variants = [("Baseline", baseline), ("Localized fallback", localized)]
    for axis, (title, episodes) in zip(axes, variants):
        for _, row in periods.iterrows():
            axis.barh(2, row["end"] - row["start"], left=row["start"], height=0.35, color="#d9d9d9", edgecolor="#888888")
        for _, row in episodes.iterrows():
            color = "#c1666b" if row["primary_family"] == "process_saturation" else "#2d6a4f"
            axis.barh(1, row["event_end"] - row["event_start"], left=row["event_start"], height=0.4, color=color, edgecolor="#333333")
        unavailability = ot_points[ot_points["is_unavailability"]]
        other_points = ot_points[~ot_points["is_unavailability"]]
        axis.scatter(unavailability["date"], [0] * len(unavailability), marker="v", s=70, color="#e76f51", label="OT indisponibilidad")
        axis.scatter(other_points["date"], [0] * len(other_points), marker="o", s=40, color="#1d3557", label="OT mantenimiento")
        axis.set_yticks([0, 1, 2])
        axis.set_yticklabels(["OT", "Episodios", "Tabla incidencias"])
        axis.set_title(title)
        axis.grid(axis="x", linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper left", ncols=3)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Incidentes en el tiempo y mantenimiento reportado (2024-02 a 2024-10)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _merge_touching_ranges(frame: pd.DataFrame, start_col: str, end_col: str, gap_hours: float = 6.0) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[start_col, end_col])
    rows = frame.sort_values(start_col)[[start_col, end_col]].to_dict("records")
    merged = [rows[0].copy()]
    gap = pd.Timedelta(hours=gap_hours)
    for row in rows[1:]:
        current = merged[-1]
        if row[start_col] <= current[end_col] + gap:
            current[end_col] = max(current[end_col], row[end_col])
        else:
            merged.append(row.copy())
    return pd.DataFrame(merged)


def plot_unavailability(periods: pd.DataFrame, ot_points: pd.DataFrame, localized: pd.DataFrame, windows: pd.DataFrame, output_path: Path) -> None:
    anomalous = windows[windows["semantic_status"] == "ANOMALOUS"].copy()
    anomalous_ranges = _merge_touching_ranges(anomalous.rename(columns={"start_time": "start", "end_time": "end"}), "start", "end", gap_hours=2.0)
    predicted = _merge_touching_ranges(localized.rename(columns={"event_start": "start", "event_end": "end"}), "start", "end", gap_hours=6.0)
    reported = periods.copy()
    reported = reported.rename(columns={"start": "start", "end": "end"})
    ot_unavailability = ot_points[ot_points["is_unavailability"]]

    fig, axis = plt.subplots(figsize=(18, 6), constrained_layout=True)
    for _, row in anomalous_ranges.iterrows():
        axis.barh(2, row["end"] - row["start"], left=row["start"], height=0.35, color="#f4a261", edgecolor="#b56576")
    for _, row in predicted.iterrows():
        axis.barh(1, row["end"] - row["start"], left=row["start"], height=0.35, color="#e76f51", edgecolor="#7f5539")
    for _, row in reported.iterrows():
        axis.barh(0, row["end"] - row["start"], left=row["start"], height=0.35, color="#adb5bd", edgecolor="#495057")
    axis.scatter(ot_unavailability["date"], [-0.55] * len(ot_unavailability), marker="v", s=70, color="#1d3557")
    axis.set_yticks([-0.55, 0, 1, 2])
    axis.set_yticklabels(["OT no servicio", "Tabla incidencias", "Episodios localized", "Ventanas anómalas"])
    axis.set_title("Indisponibilidades y periodos degradados")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_monthly_summary(periods: pd.DataFrame, baseline: pd.DataFrame, localized: pd.DataFrame, output_path: Path) -> None:
    months = pd.period_range(FOCUS_START, FOCUS_END, freq="M")

    def summarize(episodes: pd.DataFrame, label: str) -> pd.DataFrame:
        rows = []
        for month in months:
            start = month.start_time
            end = month.end_time
            overlap = episodes[(episodes.event_start <= end) & (episodes.event_end >= start)].copy()
            episode_count = len(overlap)
            active_hours = 0.0
            for _, row in overlap.iterrows():
                active_start = max(row["event_start"], start)
                active_end = min(row["event_end"], end)
                active_hours += max((active_end - active_start).total_seconds(), 0.0) / 3600.0
            rows.append({"month": month.to_timestamp(), "variant": label, "episode_count": episode_count, "active_hours": active_hours})
        return pd.DataFrame(rows)

    frame = pd.concat([summarize(baseline, "Baseline"), summarize(localized, "Localized fallback")], ignore_index=True)
    maintenance_rows = []
    for month in months:
        start = month.start_time
        end = month.end_time
        overlap = periods[(periods.start <= end) & (periods.end >= start)]
        maintenance_rows.append({"month": month.to_timestamp(), "maintenance_periods": len(overlap)})
    maintenance = pd.DataFrame(maintenance_rows)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True, constrained_layout=True)
    for variant, color in [("Baseline", "#457b9d"), ("Localized fallback", "#e76f51")]:
        subset = frame[frame["variant"] == variant]
        axes[0].plot(subset["month"], subset["episode_count"], marker="o", linewidth=2, label=variant, color=color)
        axes[1].plot(subset["month"], subset["active_hours"], marker="o", linewidth=2, label=variant, color=color)
    axes[0].bar(maintenance["month"], maintenance["maintenance_periods"], width=20, color="#ced4da", alpha=0.5, label="Periodos mantenimiento")
    axes[0].set_title("Carga mensual de episodios frente a mantenimiento")
    axes[0].set_ylabel("Episodios")
    axes[1].set_ylabel("Horas activas")
    axes[1].set_title("Horas anómalas por mes")
    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(periods: pd.DataFrame, ot_points: pd.DataFrame, baseline: pd.DataFrame, localized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, episodes in [("baseline", baseline), ("localized_fallback", localized)]:
        period_hits = sum(bool(((episodes.event_start <= row.end) & (episodes.event_end >= row.start)).any()) for row in periods.itertuples())
        point_hits = sum(bool(((episodes.event_start <= row.date + pd.Timedelta(hours=24)) & (episodes.event_end >= row.date - pd.Timedelta(hours=24))).any()) for row in ot_points.itertuples())
        rows.append(
            {
                "variant": name,
                "episode_count": int(len(episodes)),
                "median_duration_hours": float(episodes["duration_hours"].median()) if not episodes.empty else 0.0,
                "maintenance_period_hits": int(period_hits),
                "maintenance_period_total": int(len(periods)),
                "ot_point_hits_pm24h": int(point_hits),
                "ot_point_total": int(len(ot_points)),
            }
        )
    return pd.DataFrame(rows)


def write_html(summary: pd.DataFrame, output_path: Path) -> None:
    table_html = summary.to_html(index=False, float_format=lambda value: f"{value:.1f}")
    output_path.write_text(
        f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Comparativa temporal de incidentes</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 24px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Comparativa temporal de incidentes e indisponibilidades</h1>
  <p>Horizonte analizado: del 1 de febrero de 2024 al 30 de abril de 2026.</p>
  {table_html}
  <h2>Línea temporal comparativa</h2>
  <img src="incident_timeline_comparison.png" alt="Línea temporal comparativa">
  <h2>Indisponibilidades</h2>
  <img src="unavailability_timeline.png" alt="Indisponibilidades">
  <h2>Resumen mensual</h2>
  <img src="monthly_burden_comparison.png" alt="Resumen mensual">
</body>
</html>
""".strip()
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate incident timeline comparison report for probe or full outputs.")
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS), default=DEFAULT_DATASET)
    args = parser.parse_args()

    root_dir = DATASET_ROOTS[args.dataset]
    output_dir = root_dir / "comparison_report"
    baseline_dir = root_dir / "Model_D_episodes_baseline"
    localized_dir = root_dir / "Model_D_episodes_localized_fallback"

    output_dir.mkdir(parents=True, exist_ok=True)
    periods = load_maintenance_periods()
    ot_points = load_ot_points()
    baseline = load_episodes(baseline_dir)
    localized = load_episodes(localized_dir)
    localized_windows = load_windows(localized_dir)

    plot_episode_timeline(periods, ot_points, baseline, localized, output_dir / "incident_timeline_comparison.png")
    plot_unavailability(periods, ot_points, localized, localized_windows, output_dir / "unavailability_timeline.png")
    plot_monthly_summary(periods, baseline, localized, output_dir / "monthly_burden_comparison.png")

    summary = build_summary(periods, ot_points, baseline, localized)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    write_html(summary, output_dir / "incident_timeline_report.html")

    print(f"Report written to: {output_dir}")


if __name__ == "__main__":
    main()
