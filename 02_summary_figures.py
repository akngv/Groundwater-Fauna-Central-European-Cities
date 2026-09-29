"""
Summary Figures for the iterative Feature Elimination Results
================================================================

Reads the "MODEL_*.csv" and "PERIM_*.csv" files produced by
01_model_classification.py (one subfolder per region under
--results-dir/tables/) and generates the following figures per region:

- "line_diagram_cv_mean_by_model_<region>.pdf"
    CV accuracy per model as a function of the number of predictors.
- "perm_imp_matrix_<region>.pdf"
    Grid of permutation-importance bar charts, one panel per elimination step.
- "accuracy_matrix_<region>.pdf"
    Grid of CV-accuracy bar charts per model, one panel per elimination step.


...and the following combined overviews across all regions:

- "line_diagram_cv_mean_all_regions.pdf"
    2x2 grid of the per-region line diagrams above.
- "elimination_overview_all_regions.pdf"
    2x2 grid combining model accuracy (line) and permutation importance
    (bars) across the elimination steps, with the best configuration
    highlighted.

Run this after 01_model_classification.py, e.g.:
    python 02_summary_figures.py --results-dir results --regions AGG HAL VIE MUC
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = Path("results")
REGIONS = ["AGG", "HAL", "VIE", "MUC"]  # display/plot order for the 2x2 grids
GRID_LETTERS = ["a", "b", "c", "d"]
REGION_DISPLAY_NAMES = {
    "AGG": "Aggregated data (AGG)",
    "HAL": "Halle (HAL)",
    "VIE": "Vienna (VIE)",
    "MUC": "Munich (MUC)",
}

DUMMY_STRATEGIES = ["most_frequent"]

# Fixed axis ranges (set to None to auto-scale from the data instead).
ACCURACY_YLIM = (0.450, 1.000)
IMPORTANCE_YLIM = (-0.15, 0.40)

# Okabe-Ito orange - colorblind-friendly, consistently used for "best configuration"
BEST_COLOR = "#E69F00"
EXCLUDED_COLOR = "grey"
INCLUDED_COLOR = "black"

# Short labels used on plot axes for specific features, keyed by the code
# that follows the "VAL_" prefix (e.g. "VAL_NO3" -> "NO3" -> "NO₃⁻"). Any other
# feature falls back to its code with the "VAL_" prefix stripped.
FEATURE_LABELS = {
   "K": "$K$",
   "CHL": "Cl⁻",
   "PO4": "PO₄³⁻",
   "NO3": "NO₃⁻",
   "NH4": "NH₄⁺",

}

# Full names for the glossary box in the permutation-importance matrix figure.
PARAM_GLOSSARY = {
    "Cl⁻": "Chloride",
    "DSW": "Distance to surface water",
    "GWD": "Depth to groundwater",
    "K": "Hydraulic conductivity",
    "SUS": "Surface sealing",
    "NH₄⁺": "Ammonium",
    "NO₃⁻": "Nitrate",
    "DOC": "Dissolved organic carbon",
    "DO": "Dissolved oxygen",
    "PO₄³⁻": "Phosphate",
    "TMP": "Groundwater temperature",
}

CONFIG_TAG_PATTERN = re.compile(r"_C(\d+)_")  # extracts the step number from a CONFIG tag


def relabel_feature(feature) -> str:
    """Strips the "VAL_" prefix and substitutes known codes with their short label.

    Returns "-" for missing values (e.g. a step for which no feature was removed).
    """
    if pd.isna(feature):
        return "-"
    label = str(feature).replace("VAL_", "")
    for code, short in FEATURE_LABELS.items():
        label = label.replace(code, short)
    return label


# ----------------------------------------------------------------------------
# Loading & preparing the results
# ----------------------------------------------------------------------------
def load_region_results(region_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and concatenates all MODEL_*.csv and PERIM_*.csv files in a region folder."""
    model_files = sorted(region_dir.glob("MODEL_*.csv"))
    perim_files = sorted(region_dir.glob("PERIM_*.csv"))

    if not model_files:
        raise FileNotFoundError(f"No 'MODEL_*.csv' files found in {region_dir}")

    mod = pd.concat([pd.read_csv(f) for f in model_files], ignore_index=True)

    perim_parts = []
    for f in perim_files:
        tag = f.stem.removeprefix("PERIM_")
        perim_parts.append(pd.read_csv(f).assign(CONFIG=tag))
    pim = (pd.concat(perim_parts, ignore_index=True) if perim_parts
           else pd.DataFrame(columns=["Feature", "mean", "min", "max", "CONFIG"]))

    return mod, pim


def step_from_tag(tag: str) -> int:
    """Extracts the elimination-step number from a CONFIG tag like 'EVT_HAL_CS2_C03_CV05'."""
    match = CONFIG_TAG_PATTERN.search(tag)
    return int(match.group(1)) if match else -1


def ordered_config_tags(mod: pd.DataFrame) -> list:
    """All CONFIG tags for a region, sorted by elimination-step number."""
    tags = mod["CONFIG"].unique().tolist()
    return sorted(tags, key=step_from_tag)


def find_relevant_models(res: pd.DataFrame) -> pd.DataFrame:
    """Rows of models that outperform the best dummy-classifier baseline."""
    to_beat = res.loc[res.Model.isin(DUMMY_STRATEGIES), "CV_Mean"].max()
    mask = ~res.Model.isin(DUMMY_STRATEGIES) & (res.CV_Mean > to_beat)
    return res.loc[mask, ["Model", "CV_Mean"]]


def n_predictors(cell) -> int:
    """Number of predictors listed in a MODEL_*.csv 'predictors' cell (a stringified list)."""
    if pd.isna(cell):
        return 0
    return len(str(cell).strip("[]").split(",")) if str(cell).strip("[]") else 0


def get_color_cycle(n: int) -> list:
    """Returns n visually distinct colors (tab10, or tab20 if more than 10 are needed)."""
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return [cmap(i % cmap.N) for i in range(n)]


def config_summary(mod: pd.DataFrame) -> pd.DataFrame:
    """Per-CONFIG summary: mean and best CV accuracy among the relevant (non-dummy) models."""
    summary = pd.DataFrame(index=mod["CONFIG"].unique(), columns=["mean_CVmean", "best_CVmean"], dtype=float)
    for tag in summary.index:
        relevant = find_relevant_models(mod.loc[mod.CONFIG == tag])
        if len(relevant) > 0:
            summary.loc[tag, "mean_CVmean"] = relevant.CV_Mean.mean()
            summary.loc[tag, "best_CVmean"] = relevant.CV_Mean.max()
    return summary


def warn_if_outside_ylim(values, limits, quantity: str, region: str) -> None:
    """Warns if values fall outside a fixed axis range, since they would be cut off in the figures."""
    if limits is None:
        return
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return
    v_min, v_max = values.min(), values.max()
    if v_min < limits[0] or v_max > limits[1]:
        warnings.warn(
            f"{region}: {quantity} ranges from {v_min:.3f} to {v_max:.3f}, which is outside the "
            f"fixed axis range {limits}. These values will be cut off in the figures - "
            f"adjust the limits in the configuration section (or set them to None).",
            stacklevel=2,
        )


def best_step_indices(df: pd.DataFrame) -> list:
    """0-based positions of the best (max best_CVmean) step(s); ties return all of them.

    Returns an empty list if no step has a valid best_CVmean (e.g. no model ever
    outperformed the dummy baseline for this region).
    """
    values = df["best_CVmean"].values
    if np.all(np.isnan(values)):
        return []
    best_value = np.nanmax(values)
    return np.where(np.isclose(values, best_value, equal_nan=False))[0].tolist()


# ----------------------------------------------------------------------------
# Per-region figures
# ----------------------------------------------------------------------------
def plot_line_diagram(mod: pd.DataFrame, region: str, fig_dir: Path) -> pd.DataFrame:
    """CV accuracy per model vs. number of predictors, one line per model.

    Returns the prepared long-format DataFrame (used again for the combined
    2x2 overview) so it doesn't need to be recomputed.
    """
    d = mod.loc[~mod["Model"].isin(DUMMY_STRATEGIES)].copy()
    d["n_predictors"] = d["predictors"].apply(n_predictors)

    config_order = (d[["CONFIG", "n_predictors"]].drop_duplicates(subset="CONFIG")
                     .sort_values(["n_predictors", "CONFIG"]).reset_index(drop=True))
    config_order["x"] = np.arange(len(config_order))
    d = d.merge(config_order[["CONFIG", "x"]], on="CONFIG", how="left")

    models = sorted(d["Model"].unique())
    colors = get_color_cycle(len(models))

    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        sub = d.loc[d.Model == model].sort_values("x")
        plt.plot(sub["x"], sub["CV_Mean"], marker="o", color=color, label=model)

    plt.xticks(config_order["x"], config_order["n_predictors"].astype(int))
    plt.xlabel("Number of predictor parameters")
    plt.ylabel("Cross-validated mean model accuracy")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    plt.grid(axis="y", ls=":")
    plt.title(f"{region} - Cross-validated mean accuracy by model and number of parameters")
    plt.legend(frameon=True, fancybox=False, edgecolor="black", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / f"line_diagram_cv_mean_by_model_{region}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    return d


def plot_permutation_importance_matrix(pim: pd.DataFrame, tags: list, region: str, fig_dir: Path,
                                        best_indices: list) -> None:
    """Grid of permutation-importance bar charts, one panel per elimination step,
    plus a legend/glossary panel."""
    n = len(tags)
    ncols = 4
    nrows = int(np.ceil((n + 1) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    best_set = set(best_indices)

    for i, tag in enumerate(tags):
        ax = axes[i]
        imp = pim.loc[pim.CONFIG == tag].sort_values("mean")

        if imp.empty:
            # No model outperformed the dummy baseline in this step, so no
            # permutation importance was computed/saved for it.
            ax.text(0.5, 0.5, "No model beat\nthe dummy baseline", ha="center", va="center",
                    fontsize=9, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if IMPORTANCE_YLIM:
                ax.set_ylim(*IMPORTANCE_YLIM)
            ax.set_title(f"Step {i} (n=0)", fontsize=10)
            if i in best_set:
                for spine in ax.spines.values():
                    spine.set_linewidth(3)
                    spine.set_edgecolor(BEST_COLOR)
            continue

        xlabels = [relabel_feature(f) for f in imp["Feature"]]
        xpos = np.arange(len(imp))

        ax.bar(xpos, imp["mean"], color="#9ecae1", alpha=0.8, zorder=1)
        ax.vlines(xpos, imp["min"], imp["max"], color="lightgray", zorder=2)

        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
        if IMPORTANCE_YLIM:
            ax.set_ylim(*IMPORTANCE_YLIM)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"Step {i} (n={len(imp)})", fontsize=10)
        ax.grid(axis="y", ls=":")

        if i in best_set:
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_edgecolor(BEST_COLOR)

    legend_ax = axes[n]
    legend_ax.axis("off")
    legend_elements = [
        Patch(facecolor="#9ecae1", edgecolor="none", label="Mean permutation importance"),
        Line2D([0], [0], color="lightgray", lw=3, label="Min - max range"),
        Patch(facecolor="none", edgecolor=BEST_COLOR, linewidth=3, label="Best configuration"),
    ]
    legend_ax.legend(handles=legend_elements, loc="upper left", frameon=True, fancybox=False,
                      edgecolor="black", fontsize=9, bbox_to_anchor=(0, 1))
    glossary_text = "\n".join(f"{abbr} = {full}" for abbr, full in PARAM_GLOSSARY.items())
    legend_ax.text(0, 0.55, glossary_text, transform=legend_ax.transAxes,
                    fontsize=7.5, va="top", family="monospace")

    for j in range(n + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{region} - Permutation importance per configuration", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / f"perm_imp_matrix_{region}.pdf", dpi=300)
    plt.close(fig)


def plot_accuracy_matrix(mod: pd.DataFrame, tags: list, region: str, fig_dir: Path,
                          best_indices: list) -> None:
    """Grid of CV-accuracy bar charts (per model, incl. min-max range), one panel per
    elimination step, with the most-frequent-class baseline as a reference line."""
    n = len(tags)
    ncols = 4
    nrows = int(np.ceil((n + 1) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    best_set = set(best_indices)

    for i, tag in enumerate(tags):
        ax = axes[i]
        res = mod.loc[mod.CONFIG == tag].sort_values("CV_Mean")

        most_freq_row = res.loc[res.Model == "most_frequent", "CV_Mean"]
        most_freq = most_freq_row.values[0] if len(most_freq_row) else np.nan

        bars = res.loc[~res.Model.isin(DUMMY_STRATEGIES)]
        y_pos = np.arange(len(bars))

        ax.barh(y_pos, bars["CV_Mean"], height=0.7, color="0.6", zorder=1)
        ax.hlines(y=y_pos, xmin=bars["CV_Min"], xmax=bars["CV_Max"], color="k", zorder=2)
        ax.scatter(bars["CV_Min"], y_pos, marker="|", c="k", zorder=3)
        ax.scatter(bars["CV_Max"], y_pos, marker="|", c="k", zorder=3)

        if not np.isnan(most_freq):
            ax.axvline(most_freq, color="red", ls=":", zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(bars["Model"], fontsize=7)
        if ACCURACY_YLIM:
            ax.set_xlim(*ACCURACY_YLIM)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(f"Step {i} (n={len(bars)})", fontsize=10)
        ax.grid(axis="x", ls=":")

        if i in best_set:
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_edgecolor(BEST_COLOR)

    legend_ax = axes[n]
    legend_ax.axis("off")
    legend_elements = [
        Patch(facecolor="0.6", edgecolor="none", label="Model accuracy (mean)"),
        Line2D([0], [0], color="k", lw=2, label="Min - max range"),
        Line2D([0], [0], color="red", ls=":", lw=2, label="Most frequent value"),
        Patch(facecolor="none", edgecolor=BEST_COLOR, linewidth=3, label="Best configuration"),
    ]
    legend_ax.legend(handles=legend_elements, loc="center", frameon=True, fancybox=False,
                      edgecolor="black", fontsize=9)

    for j in range(n + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{region} - Cross-validated accuracy per model and configuration", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / f"accuracy_matrix_{region}.pdf", dpi=300)
    plt.close(fig)


def build_elimination_overview(mod: pd.DataFrame, pim: pd.DataFrame, tags: list) -> pd.DataFrame:
    """Per-step table: mean/best CV accuracy plus the least important feature removed next."""
    summary = config_summary(mod).loc[tags]

    rows = []
    for tag in tags:
        subset = pim.loc[pim.CONFIG == tag]
        if subset.empty:
            # No model outperformed the dummy baseline in this step (no permutation
            # importance was computed/saved for it) - record NaNs instead of failing.
            rows.append({
                "CONFIG": tag,
                "removed_feature": np.nan,
                "removed_feature_importance": np.nan,
                "removed_feature_importance_min": np.nan,
                "removed_feature_importance_max": np.nan,
            })
            continue
        least_important = subset.sort_values(by="mean").iloc[0]
        rows.append({
            "CONFIG": tag,
            "removed_feature": least_important["Feature"],
            "removed_feature_importance": least_important["mean"],
            "removed_feature_importance_min": least_important["min"],
            "removed_feature_importance_max": least_important["max"],
        })
    overview = summary.join(pd.DataFrame(rows).set_index("CONFIG"))

    most_frequent_value = mod.loc[(mod.CONFIG == tags[0]) & (mod.Model == "most_frequent"), "CV_Mean"].values[0]
    overview.loc["MOST_FREQUENT"] = [most_frequent_value, most_frequent_value, np.nan, np.nan, np.nan, np.nan]

    # Shift by one row so that each step is labelled with the feature removed BEFORE it,
    # i.e. at the end of step i-1 - the feature that is absent from step i onwards.
    # Step 0 still contains all features and gets "-"; the appended MOST_FREQUENT row
    # receives the feature removed at the end of the last step.
    shift_cols = ["removed_feature", "removed_feature_importance",
                  "removed_feature_importance_min", "removed_feature_importance_max"]
    overview[shift_cols] = overview[shift_cols].shift(1)
    overview["removed_feature"] = overview["removed_feature"].astype(object)
    overview.iloc[0, overview.columns.get_loc("removed_feature")] = "-"

    return overview


# ----------------------------------------------------------------------------
# Combined (2x2) overviews across all regions
# ----------------------------------------------------------------------------
def plot_line_diagram_grid(line_data: dict, output_dir: Path) -> None:
    """2x2 grid of the per-region CV-accuracy line diagrams, with consistent colors per model."""
    all_models = sorted(set().union(*[set(d["Model"].unique()) for d in line_data.values()]))
    color_map = dict(zip(all_models, get_color_cycle(len(all_models))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    y_min = min(d["CV_Mean"].min() for d in line_data.values())
    y_max = max(d["CV_Mean"].max() for d in line_data.values())

    handles_by_model = {}
    for ax, region, letter in zip(axes, REGIONS, GRID_LETTERS):
        d = line_data[region]
        config_order = (d[["CONFIG", "n_predictors", "x"]].drop_duplicates(subset="CONFIG").sort_values("x"))

        for model in sorted(d["Model"].unique()):
            sub = d.loc[d.Model == model].sort_values("x")
            line, = ax.plot(sub["x"], sub["CV_Mean"], marker="o", color=color_map[model], label=model)
            handles_by_model[model] = line

        ax.set_xticks(config_order["x"])
        ax.set_xticklabels(config_order["n_predictors"].astype(int))
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(f"({letter}) {REGION_DISPLAY_NAMES.get(region, region)}")
        ax.grid(axis="y", ls=":")
        ax.set_xlabel("Number of predictor parameters")
        if letter in ("a", "c"):
            ax.set_ylabel("Cross-validated mean model accuracy")

    fig.suptitle("Cross-validated mean accuracy by model and number of parameters", fontsize=14)
    ordered_models = sorted(handles_by_model.keys())
    fig.legend([handles_by_model[m] for m in ordered_models], ordered_models,
               loc="lower center", ncol=min(len(ordered_models), 6),
               frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(output_dir / "line_diagram_cv_mean_all_regions.pdf", dpi=300)
    plt.close(fig)


def plot_elimination_overview_grid(overviews: dict, output_dir: Path) -> None:
    """2x2 grid combining, per region: best/mean model accuracy (lines) and the
    permutation importance of the next-removed feature (bars), with the best
    configuration marked and excluded parameters greyed out on the x-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    y1_min = min(df["mean_CVmean"].min() for df in overviews.values())
    y1_max = max(df["best_CVmean"].max() for df in overviews.values())
    y2_min = min(df["removed_feature_importance_min"].min() for df in overviews.values())
    y2_max = max(df["removed_feature_importance_max"].max() for df in overviews.values())
    if ACCURACY_YLIM:
        y1_min, y1_max = ACCURACY_YLIM
    if IMPORTANCE_YLIM:
        y2_min, y2_max = IMPORTANCE_YLIM

    l1 = l2 = l3 = best_marker = vlines = None

    for ax, region, letter in zip(axes, REGIONS, GRID_LETTERS):
        df = overviews[region]
        x = np.arange(len(df))
        best_idx = best_step_indices(df)
        most_frequent_value = df["mean_CVmean"].iloc[-1]

        ax2 = ax.twinx()
        ax2.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

        ax2.bar(x, df["removed_feature_importance"], color="#9ecae1", alpha=0.8, zorder=1)
        vlines = ax2.vlines(x, df["removed_feature_importance_min"], df["removed_feature_importance_max"],
                             color="lightgray", zorder=2)

        l1, = ax.plot(x, df["best_CVmean"], "k-", marker="o", zorder=3)
        l2, = ax.plot(x, df["mean_CVmean"], linestyle="--", marker="o", color="dimgray", zorder=3)
        l3 = ax.hlines(y=most_frequent_value, xmin=-0.5, xmax=len(df) - 0.5,
                        colors="red", linestyles="dotted", zorder=3)

        # best_idx is empty if no step in this region ever beat the dummy baseline;
        # in that case nothing is highlighted and all labels stay the default color.
        best_idx_arr = np.asarray(best_idx, dtype=int)
        boundary = max(best_idx) if best_idx else -1
        best_marker = ax.scatter(x[best_idx_arr], df["best_CVmean"].iloc[best_idx_arr],
                                  marker="D", s=90, facecolor=BEST_COLOR, edgecolor="black",
                                  linewidth=0.8, zorder=5, label="Best configuration")

        ax.set_ylim(y1_min, y1_max)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"({letter}) {REGION_DISPLAY_NAMES.get(region, region)}")
        ax.set_xticks(x)
        ax.set_xlim(-0.5, len(df) - 0.5)
        ax.set_xticklabels(df["removed_feature"].apply(relabel_feature), rotation=45)

        for i, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(EXCLUDED_COLOR if i <= boundary else INCLUDED_COLOR)

        ax.grid(axis="y", ls=":")
        ax2.set_ylim(y2_min, y2_max)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax2.tick_params(axis="y", colors="#3182bd")

        # Avoid duplicating y-axis labels: left column gets the accuracy label,
        # right column gets the permutation-importance label.
        ax.set_ylabel("Model accuracy" if letter in ("a", "c") else "")
        ax2.set_ylabel("Permutation importance" if letter in ("b", "d") else "", color="#3182bd")
        ax.set_xlabel("\u27f6    \u27f6   Progressively removed parameters   \u27f6    \u27f6"
                       if letter in ("c", "d") else "")

    fig.legend([l1, l2, l3, best_marker, vlines],
               ["Best model", "Mean model", "Most frequent value", "Best configuration", "Min-max"],
               loc="lower center", ncol=5, frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "elimination_overview_all_regions.pdf", dpi=300)
    fig.savefig(output_dir / "elimination_overview_all_regions.svg", dpi=300)

    plt.close(fig)


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
def main() -> None:
    global REGIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                         help="Directory produced by 01_model_classification.py "
                              "(expects <results-dir>/tables/<REGION>/)")
    parser.add_argument("--regions", nargs="+", default=REGIONS, help="Regions to process, in plot order")
    args = parser.parse_args()

    REGIONS = args.regions

    tables_dir = args.results_dir / "tables"
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    line_data = {}
    overviews = {}

    for region in REGIONS:
        print(f"\n===== {region} =====")
        region_fig_dir = figures_dir / region
        region_fig_dir.mkdir(parents=True, exist_ok=True)

        mod, pim = load_region_results(tables_dir / region)
        tags = ordered_config_tags(mod)

        non_dummy = mod.loc[~mod.Model.isin(DUMMY_STRATEGIES)]
        dummy_mean = mod.loc[mod.Model.isin(DUMMY_STRATEGIES), "CV_Mean"]
        warn_if_outside_ylim(pd.concat([non_dummy["CV_Min"], non_dummy["CV_Max"], dummy_mean]),
                             ACCURACY_YLIM, "CV accuracy", region)
        warn_if_outside_ylim(pd.concat([pim["min"], pim["max"]]),
                             IMPORTANCE_YLIM, "Permutation importance", region)

        summary = config_summary(mod).loc[tags]
        best_indices = best_step_indices(summary)

        line_data[region] = plot_line_diagram(mod, region, region_fig_dir)
        plot_permutation_importance_matrix(pim, tags, region, region_fig_dir, best_indices)
        plot_accuracy_matrix(mod, tags, region, region_fig_dir, best_indices)

        overview = build_elimination_overview(mod, pim, tags)
        overviews[region] = overview

    plot_line_diagram_grid(line_data, figures_dir)
    plot_elimination_overview_grid(overviews, figures_dir)

    print(f"\nDone. Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
