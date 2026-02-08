#!/usr/bin/env python3
"""
Academic Data Visualization Dashboard
=========================================================
Generates publication-quality charts from generated academic data.

Usage:
    python visualize_academic_data.py
    python visualize_academic_data.py --data-dir ./custom_data
    python visualize_academic_data.py --output-dir ./charts

Author: Wington Brito
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


# ── Theme ─────────────────────────────────────────────────────────────────

THEME_COLORS = {
    "primary": "#1B3A5C",      # dark navy
    "secondary": "#2E86AB",    # bright blue
    "accent": "#F18F01",       # orange
    "success": "#2CA58D",      # teal/green
    "danger": "#C1292E",       # red
    "warning": "#F4D35E",      # yellow
    "light": "#E8EEF2",       # light gray-blue
    "text": "#2C3E50",         # dark text
}

SEVERITY_COLORS = {
    "critical": THEME_COLORS["danger"],
    "high": THEME_COLORS["accent"],
    "medium": THEME_COLORS["warning"],
    "low": THEME_COLORS["success"],
}

FLAG_COLORS = [
    THEME_COLORS["danger"],
    THEME_COLORS["accent"],
    THEME_COLORS["secondary"],
    THEME_COLORS["success"],
    "#8E44AD",
    "#34495E",
    "#E74C3C",
]


def apply_theme():
    """Apply theme styling to matplotlib."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.facecolor": "#FAFBFC",
        "axes.edgecolor": "#DEE2E6",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#CED4DA",
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


# ── Chart Builders ────────────────────────────────────────────────────────

def chart_enrollment_growth(summary: dict, output_dir: Path):
    """Dual-axis enrollment + student growth with trend line."""
    snapshots = summary["snapshots"]
    semesters = [s["semester_name"] for s in snapshots]
    enrollments = [s["total_enrollments"] for s in snapshots]
    students = [s["unique_students"] for s in snapshots]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Enrollment bars
    bars = ax1.bar(
        range(len(semesters)), enrollments,
        color=THEME_COLORS["secondary"], alpha=0.85, label="Total Enrollments",
        edgecolor="white", linewidth=0.5,
    )
    ax1.set_xlabel("Semester")
    ax1.set_ylabel("Total Enrollments", color=THEME_COLORS["secondary"])
    ax1.set_xticks(range(len(semesters)))
    ax1.set_xticklabels(semesters, rotation=30, ha="right")
    ax1.tick_params(axis="y", labelcolor=THEME_COLORS["secondary"])

    # Add value labels on bars
    for bar, val in zip(bars, enrollments):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{val:,}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=THEME_COLORS["text"],
        )

    # Student line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(semesters)), students,
        color=THEME_COLORS["accent"], marker="o", linewidth=2.5,
        markersize=8, label="Unique Students", zorder=5,
    )
    ax2.set_ylabel("Unique Students", color=THEME_COLORS["accent"])
    ax2.tick_params(axis="y", labelcolor=THEME_COLORS["accent"])

    # Growth annotation
    trends = summary.get("trends", {})
    growth_text = (
        f"Overall Growth: +{trends.get('overall_growth_pct', 0):.0f}%\n"
        f"Avg Semester: +{trends.get('mean_enrollment_increase_pct', 0):.1f}%"
    )
    ax1.text(
        0.02, 0.95, growth_text, transform=ax1.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=THEME_COLORS["light"], alpha=0.9),
    )

    fig.suptitle(
        "Enrollment Growth Trend",
        fontsize=16, fontweight="bold", color=THEME_COLORS["primary"], y=1.02,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    path = output_dir / "enrollment_growth.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_gpa_trend(summary: dict, output_dir: Path):
    """GPA trend with visual band."""
    gpa_data = summary.get("trends", {}).get("gpa_trend", [])
    if not gpa_data:
        return None

    semesters = [g["semester"] for g in gpa_data]
    gpas = [g["mean_gpa"] for g in gpa_data]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        range(len(semesters)), gpas,
        color=THEME_COLORS["primary"], marker="s", linewidth=2.5,
        markersize=8, zorder=5,
    )
    ax.fill_between(
        range(len(semesters)),
        [g - 0.05 for g in gpas],
        [g + 0.05 for g in gpas],
        alpha=0.15, color=THEME_COLORS["secondary"],
    )

    # Annotate each point
    for i, gpa in enumerate(gpas):
        ax.annotate(
            f"{gpa:.2f}", (i, gpa),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9, fontweight="bold",
        )

    # Reference lines
    ax.axhline(y=2.0, color=THEME_COLORS["danger"], linestyle="--", alpha=0.6, label="Probation (2.0)")
    ax.axhline(y=3.0, color=THEME_COLORS["success"], linestyle="--", alpha=0.6, label="Dean's List (3.0)")

    ax.set_xticks(range(len(semesters)))
    ax.set_xticklabels(semesters, rotation=30, ha="right")
    ax.set_ylabel("Mean GPA")
    ax.set_ylim(1.5, 4.0)
    ax.legend(loc="lower left", framealpha=0.9)

    fig.suptitle(
        "Mean GPA Trend",
        fontsize=16, fontweight="bold", color=THEME_COLORS["primary"], y=1.02,
    )

    path = output_dir / "gpa_trend.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_registration_flags(enrollments_path: Path, gpa_trends_path: Path, output_dir: Path):
    """Stacked bar of registration flags by semester."""
    if not enrollments_path.exists():
        return None
    df = pd.read_csv(enrollments_path)

    # Parse flags from JSON strings
    flag_records = []
    for _, row in df.iterrows():
        flags_raw = row.get("flags", "[]")
        if pd.isna(flags_raw) or flags_raw == "[]":
            continue
        try:
            flags = json.loads(flags_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for flag in flags:
            flag_records.append({
                "semester": row["semester_name"],
                "flag": flag.replace("RegistrationFlag.", ""),
            })

    # Add GPA decline flags from trends
    if gpa_trends_path.exists():
        tdf = pd.read_csv(gpa_trends_path)
        for _, row in tdf.iterrows():
            flags_raw = row.get("flags", "[]")
            if pd.isna(flags_raw) or flags_raw == "[]":
                continue
            try:
                flags = json.loads(flags_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            for flag in flags:
                flag_records.append({
                    "semester": row["semester_name"],
                    "flag": flag.replace("RegistrationFlag.", ""),
                })

    if not flag_records:
        return None

    flag_df = pd.DataFrame(flag_records)
    pivot = flag_df.groupby(["semester", "flag"]).size().unstack(fill_value=0)

    # Sort semesters chronologically
    sem_order = sorted(pivot.index, key=lambda s: (int(s.split()[-1]), 0 if "Spring" in s else 1))
    pivot = pivot.reindex(sem_order)

    fig, ax = plt.subplots(figsize=(13, 6))
    pivot.plot(
        kind="bar", stacked=True, ax=ax,
        color=FLAG_COLORS[:len(pivot.columns)],
        edgecolor="white", linewidth=0.5,
    )

    ax.set_xlabel("Semester")
    ax.set_ylabel("Flag Count")
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.legend(
        title="Flag Type", bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9, title_fontsize=10,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle(
        "Registration Anomaly Flags by Semester",
        fontsize=16, fontweight="bold", color=THEME_COLORS["primary"], y=1.02,
    )

    path = output_dir / "registration_flags.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_over_enrollment(alerts_path: Path, output_dir: Path):
    """Horizontal bar chart of over-enrollment by severity."""
    if not alerts_path.exists():
        return None
    with open(alerts_path) as f:
        alerts = json.load(f)

    if not alerts:
        return None

    # Sort by overflow_pct descending, take top 20
    alerts = sorted(alerts, key=lambda a: -a["overflow_pct"])[:20]

    labels = [f"{a['course_code']} ({a['semester_name']})" for a in alerts]
    overflow = [a["overflow_pct"] * 100 for a in alerts]
    colors = [SEVERITY_COLORS.get(a["severity"], "#999") for a in alerts]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(len(labels)), overflow, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Over-Enrollment %")
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, overflow):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", fontsize=8, fontweight="bold",
        )

    # Severity legend
    legend_handles = [
        Patch(facecolor=SEVERITY_COLORS["critical"], label="Critical (>25%)"),
        Patch(facecolor=SEVERITY_COLORS["high"], label="High (15-25%)"),
        Patch(facecolor=SEVERITY_COLORS["medium"], label="Medium (5-15%)"),
        Patch(facecolor=SEVERITY_COLORS["low"], label="Low (<5%)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    fig.suptitle(
        "Over-Enrollment Alerts (Top 20)",
        fontsize=16, fontweight="bold", color=THEME_COLORS["primary"], y=1.02,
    )

    path = output_dir / "over_enrollment.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_department_breakdown(students_path: Path, output_dir: Path):
    """Donut chart of student distribution by major."""
    if not students_path.exists():
        return None
    df = pd.read_csv(students_path)
    dept_counts = df["major"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 9))

    dept_colors = plt.cm.Set3(range(len(dept_counts)))

    wedges, texts, autotexts = ax.pie(
        dept_counts.values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * dept_counts.sum()))})",
        colors=dept_colors,
        pctdistance=0.78,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )

    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax.legend(
        wedges, dept_counts.index,
        title="Department",
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        fontsize=9,
        title_fontsize=10,
    )

    # Center text
    ax.text(0, 0, f"{dept_counts.sum()}\nStudents", ha="center", va="center",
            fontsize=16, fontweight="bold", color=THEME_COLORS["primary"])

    fig.suptitle(
        "Student Distribution by Major",
        fontsize=16, fontweight="bold", color=THEME_COLORS["primary"], y=0.98,
    )

    path = output_dir / "department_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_semester_dashboard(summary: dict, output_dir: Path):
    """Multi-panel dashboard combining key metrics."""
    snapshots = summary["snapshots"]
    semesters = [s["semester_name"] for s in snapshots]
    trends = summary.get("trends", {})

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Enrollment + Students ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    enroll = [s["total_enrollments"] for s in snapshots]
    students = [s["unique_students"] for s in snapshots]
    ax1.bar(range(len(semesters)), enroll, color=THEME_COLORS["secondary"], alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(semesters)), students, color=THEME_COLORS["accent"], marker="o", linewidth=2)
    ax1.set_title("Enrollment vs Students")
    ax1.set_xticks(range(len(semesters)))
    ax1.set_xticklabels([s.replace(" ", "\n") for s in semesters], fontsize=7)

    # ── Panel 2: GPA Trend ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    gpas = [s["mean_gpa"] for s in snapshots]
    ax2.plot(range(len(semesters)), gpas, color=THEME_COLORS["primary"], marker="s", linewidth=2)
    ax2.fill_between(range(len(semesters)), gpas, alpha=0.15, color=THEME_COLORS["secondary"])
    ax2.axhline(y=2.0, color=THEME_COLORS["danger"], linestyle="--", alpha=0.5)
    ax2.set_title("Mean GPA Trend")
    ax2.set_xticks(range(len(semesters)))
    ax2.set_xticklabels([s.replace(" ", "\n") for s in semesters], fontsize=7)
    ax2.set_ylim(2.0, 3.5)

    # ── Panel 3: Over-Enrollment Sections ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    over = [s["over_enrolled_sections"] for s in snapshots]
    overflow_seats = [s["total_overflow_seats"] for s in snapshots]
    bars = ax3.bar(range(len(semesters)), over, color=THEME_COLORS["danger"], alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(len(semesters)), overflow_seats, color=THEME_COLORS["accent"], marker="D", linewidth=2)
    ax3_twin.set_ylabel("Overflow Seats", fontsize=8, color=THEME_COLORS["accent"])
    ax3.set_title("Over-Capacity Sections")
    ax3.set_xticks(range(len(semesters)))
    ax3.set_xticklabels([s.replace(" ", "\n") for s in semesters], fontsize=7)

    # ── Panel 4: Growth Rate ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    changes = trends.get("enrollment_changes", [])
    if changes:
        periods = [f"{c['from'].split()[0][:2]}{c['from'].split()[1][2:]}\n->{c['to'].split()[0][:2]}{c['to'].split()[1][2:]}" for c in changes]
        e_change = [c["enrollment_change_pct"] for c in changes]
        s_change = [c["student_change_pct"] for c in changes]
        x = range(len(periods))
        w = 0.35
        ax4.bar([i - w/2 for i in x], e_change, w, color=THEME_COLORS["secondary"], label="Enrollment")
        ax4.bar([i + w/2 for i in x], s_change, w, color=THEME_COLORS["success"], label="Students")
        ax4.set_xticks(list(x))
        ax4.set_xticklabels(periods, fontsize=7)
        ax4.legend(fontsize=8)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax4.set_title("Semester Growth Rate")

    # ── Panel 5: Avg Class Size ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sizes = [s["avg_class_size"] for s in snapshots]
    ax5.plot(range(len(semesters)), sizes, color=THEME_COLORS["secondary"], marker="o", linewidth=2.5)
    ax5.fill_between(range(len(semesters)), sizes, alpha=0.1, color=THEME_COLORS["secondary"])
    ax5.set_title("Average Class Size")
    ax5.set_xticks(range(len(semesters)))
    ax5.set_xticklabels([s.replace(" ", "\n") for s in semesters], fontsize=7)

    # ── Panel 6: Key Metrics Summary ──────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    metrics = [
        ("Students", f"{summary['total_students']:,}"),
        ("Courses", f"{summary['total_courses']}"),
        ("Enrollments", f"{summary['total_enrollments']:,}"),
        ("Drop Rate", f"{summary['total_drops'] / max(summary['total_enrollments'], 1) * 100:.1f}%"),
        ("Flagged", f"{summary['total_flagged_enrollments']:,}"),
        ("Over-Cap Alerts", f"{summary['over_enrollment_alerts']}"),
        ("Critical Alerts", f"{summary['critical_alerts']}"),
        ("GPA Warnings", f"{summary['gpa_decline_alerts']}"),
        ("Growth", f"+{trends.get('overall_growth_pct', 0):.0f}%"),
    ]
    for i, (label, value) in enumerate(metrics):
        y = 0.92 - i * 0.105
        ax6.text(0.05, y, label, fontsize=10, fontweight="bold",
                 color=THEME_COLORS["text"], transform=ax6.transAxes)
        ax6.text(0.85, y, value, fontsize=11, fontweight="bold",
                 color=THEME_COLORS["primary"], ha="right", transform=ax6.transAxes)

    ax6.set_title("Key Metrics", pad=10)
    ax6.patch.set_facecolor(THEME_COLORS["light"])
    ax6.patch.set_alpha(0.5)

    fig.suptitle(
        "Registrar Dashboard",
        fontsize=18, fontweight="bold", color=THEME_COLORS["primary"], y=1.01,
    )

    path = output_dir / "dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Academic Data Visualizer")
    parser.add_argument(
        "--data-dir",
        default="./output",
        help="Directory with generated JSON/CSV data",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/charts",
        help="Directory to save charts",
    )
    args = parser.parse_args()

    data = Path(args.data_dir)
    out = Path(args.output_dir)

    summary_path = data / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found.")
        print("Run the generator first:")
        print("  python generate_academic_data.py --output all")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    apply_theme()

    # Load summary
    with open(summary_path) as f:
        summary = json.load(f)

    print("Generating charts...")

    charts = []
    charts.append(("Enrollment Growth", chart_enrollment_growth(summary, out)))
    charts.append(("GPA Trend", chart_gpa_trend(summary, out)))
    charts.append(("Registration Flags", chart_registration_flags(
        data / "enrollments.csv", data / "gpa_trends.csv", out,
    )))
    charts.append(("Over-Enrollment", chart_over_enrollment(
        data / "over_enrollment_alerts.json", out,
    )))
    charts.append(("Department Breakdown", chart_department_breakdown(
        data / "students.csv", out,
    )))
    charts.append(("Dashboard", chart_semester_dashboard(summary, out)))

    generated = [(n, p) for n, p in charts if p]
    print(f"\nGenerated {len(generated)} charts:")
    for name, path in generated:
        print(f"  {name:25s} -> {path}")

    print(f"\nAll charts saved to: {out}/")


if __name__ == "__main__":
    main()
