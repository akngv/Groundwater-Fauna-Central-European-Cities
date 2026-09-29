"""
Created on Tue Aug 25 14:08:30 2026


Model Classification with iterative Feature Elimination according to Permutation Importance
==============================================================================

For each region, a set of common classification models (plus dummy
baselines) is compared via GridSearchCV + cross-validation. The least
important feature (based on permutation importance across all models that
outperform the best dummy classifier) is then removed iteratively, until
only one feature remains or no model beats the dummy baseline anymore.

For each step, the following are saved:
- CSV with the CV results of all models ("MODEL_<tag>.csv")
- CSV with the permutation importance of the relevant models ("PERIM_<tag>.csv")

No figures are produced by this script — it only writes the CSV files.
Figures are generated separately from these CSVs via 02_summary_figures.py.

Expected input data: a CSV with an "ID" column (used as the row index), a
"REG" (Region) column, a "Target" column (see TARGET_COLUMN), and several feature
columns prefixed with "VAL_" (e.g. "VAL_NO3", "VAL_SUS", ...).
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Paths are kept relative to the project directory and can be overridden via
# the command line, e.g.:
#   python 01_model_classification.py --data dataset.csv --output-dir results/
DEFAULT_DATA_PATH = Path("dataset.csv")
DEFAULT_OUTPUT_DIR = Path("results")

REGIONS = ["HAL", "MUC", "VIE", "AGG"]  # "AGG" = all regions aggregated
REGION_COLUMN = "REG"
TARGET_COLUMN = "Target"
FEATURE_PREFIX = "VAL_"  # feature columns are identified by this prefix, e.g. "VAL_CHL"
CV_SPLITS_DEFAULT = 5
CV_SPLITS_AGGREGATED = 10
RANDOM_STATE = 20
DUMMY_STRATEGIES = ["most_frequent"]


# ----------------------------------------------------------------------------
# Model training & evaluation
# ----------------------------------------------------------------------------
def grid_search_with_cv(X: pd.DataFrame, y: pd.Series, cv) -> Tuple[dict, pd.DataFrame]:
    """Trains several classification models via GridSearchCV and compares them via CV.

    Returns the best models (dict: name -> pipeline) and a results table with
    CV statistics. Dummy classifiers are added as baselines.
    """
    predictors = sorted(X.columns)

    param_grids = {
        "LogReg": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"model__C": [0.1, 1.0, 10.0]},
        },
        "SVM": {
            "model": SVC(probability=True),
            "params": {"model__C": [0.1, 1, 10], "model__gamma": ["scale"]},
        },
        "kNN": {
            "model": KNeighborsClassifier(),
            "params": {"model__n_neighbors": [3, 5, 7]},
        },
        "DecTre": {
            "model": DecisionTreeClassifier(),
            "params": {"model__max_depth": [None, 3, 5, 7], "model__min_samples_split": [2, 4, 6]},
        },
        "RanFor": {
            "model": RandomForestClassifier(),
            "params": {
                "model__n_estimators": [25, 50, 100, 200],
                "model__max_depth": [None, 4, 6, 8],
                "model__min_samples_split": [2, 4, 6],
            },
        },
        "GrdBst": {
            "model": GradientBoostingClassifier(),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [None, 2, 3, 4, 5],
            },
        },
        "AdaBst": {
            "model": AdaBoostClassifier(),
            "params": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 1]},
        },
        "LDA": {"model": LinearDiscriminantAnalysis(), "params": {}},
        "MLP": {
            "model": MLPClassifier(max_iter=2000, tol=1e-3, random_state=42),
            "params": {"model__hidden_layer_sizes": [(50,), (100,)], "model__alpha": [0.0001, 0.001]},
        },
    }

    scaled_models = {"LogReg", "SVM", "kNN", "MLP", "LDA"}
    results = []
    best_models = {}

    for name, cfg in tqdm(param_grids.items(), desc="Training models"):
        try:
            steps = [("scaler", StandardScaler())] if name in scaled_models else []
            steps.append(("model", cfg["model"]))
            pipe = Pipeline(steps)

            grid = GridSearchCV(
                estimator=pipe,
                param_grid=cfg["params"],
                scoring="accuracy",
                cv=cv,
                n_jobs=1,
                verbose=0,
            )
            grid.fit(X, y)

            best_model = grid.best_estimator_
            scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")

            results.append({
                "Model": name,
                "Model_Parameters": grid.best_params_,
                "CV_Mean": scores.mean(),
                "CV_Std": scores.std(),
                "CV_Min": scores.min(),
                "CV_Max": scores.max(),
                "CV_Scores": np.round(scores, 3).tolist(),
            })
            best_models[name] = best_model
        except Exception as e:
            print(f"Error training model '{name}': {e}")

    for strategy in DUMMY_STRATEGIES:
        dummy = DummyClassifier(strategy=strategy)
        dummy.fit(X, y)
        scores = cross_val_score(dummy, X, y, cv=cv, scoring="accuracy")
        best_models[strategy] = dummy
        results.append({
            "Model": strategy,
            "CV_Mean": scores.mean(),
            "CV_Std": scores.std(),
            "CV_Min": scores.min(),
            "CV_Max": scores.max(),
            "CV_Scores": np.round(scores, 3).tolist(),
        })

    results_df = pd.DataFrame(results).sort_values(by="CV_Mean", ascending=False)
    print("\nCross-validation results:")
    print(results_df[["Model", "CV_Mean", "CV_Std", "CV_Min", "CV_Max", "Model_Parameters"]])
    results_df.insert(len(results_df.columns), "predictors", str(predictors))

    return best_models, results_df


def compute_permutation_importance(models: dict, X: pd.DataFrame, y: pd.Series, cv):
    """Computes permutation importance per model across all CV folds.

    Returns a DataFrame indexed by feature, with one column per model plus
    aggregate "mean"/"max"/"min"/"std" columns — or None if `models` is empty
    (i.e. no model outperforms the dummy baseline for this step).
    """
    feature_dfs = []
    X_reset, y_reset = X.reset_index(drop=True), y.reset_index(drop=True)

    for name, model in models.items():
        importances = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X_reset.loc[train_idx], X_reset.loc[test_idx]
            y_train, y_test = y_reset.loc[train_idx], y_reset.loc[test_idx]
            model.fit(X_train, y_train)
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            importances.append(result.importances_mean)

        feature_dfs.append(pd.DataFrame({
            "Model": name,
            "Feature": X.columns,
            "PermImp_mean": np.mean(importances, axis=0),
            "PermImp_std": np.std(importances, axis=0),
            "PermImp_min": np.min(importances, axis=0),
            "PermImp_max": np.max(importances, axis=0),
        }))

    if not feature_dfs:
        return None

    all_feat_imp = pd.concat(feature_dfs)

    mean = (all_feat_imp[["Feature", "PermImp_mean"]].groupby("Feature").mean()
            .sort_values("PermImp_mean", ascending=False).rename(columns={"PermImp_mean": "mean"}))
    maxi = all_feat_imp[["Feature", "PermImp_max"]].groupby("Feature").max().rename(columns={"PermImp_max": "max"})
    mini = all_feat_imp[["Feature", "PermImp_min"]].groupby("Feature").min().rename(columns={"PermImp_min": "min"})
    std = all_feat_imp[["Feature", "PermImp_std"]].groupby("Feature").std().rename(columns={"PermImp_std": "std"})

    pivoted = pd.pivot(all_feat_imp, values="PermImp_mean", index="Feature", columns="Model")
    imp_desc = pivoted.join(mean).join(maxi).join(mini).join(std).sort_values("mean", ascending=False)

    return imp_desc

def find_relevant_models(results_df: pd.DataFrame) -> np.ndarray:
    """Names of all models that outperform the best dummy baseline."""
    to_beat = results_df.loc[results_df.Model.isin(DUMMY_STRATEGIES), "CV_Mean"].max()
    mask = ~results_df.Model.isin(DUMMY_STRATEGIES) & (results_df.CV_Mean > to_beat)
    return results_df.loc[mask, "Model"].values


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
def run_region(EV_all: pd.DataFrame, region: str, output_dir: Path) -> None:
    """Runs iterative feature elimination for one region (or 'AGG' = all regions)."""
    print(f"\n===== RUNNING {region} =====")

    if region == "AGG":
        EV = EV_all.copy()
        cv_splits = CV_SPLITS_AGGREGATED
    else:
        EV = EV_all.loc[EV_all[REGION_COLUMN] == region].copy()
        cv_splits = CV_SPLITS_DEFAULT

    feature_columns = [c for c in EV.columns if c.startswith(FEATURE_PREFIX)]
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    res_dir = output_dir / "tables" / region
    res_dir.mkdir(parents=True, exist_ok=True)

    current_features = feature_columns.copy()
    step = 0

    while len(current_features) >= 1:
        tag = f"EVT_{region}_CS2_C{step:02d}_CV{cv_splits:02d}"
        print(f"\n--- {tag} | features: {len(current_features)} ---")

        X = EV[current_features]
        y = EV[TARGET_COLUMN]

        best_models, results_df = grid_search_with_cv(X, y, cv)

        relevant_names = find_relevant_models(results_df)
        relevant_models = {name: best_models[name] for name in relevant_names if name in best_models}

        perm_importance = compute_permutation_importance(relevant_models, X, y, cv)

        results_df.insert(0, "CONFIG", tag)
        results_df.to_csv(res_dir / f"MODEL_{tag}.csv", index=False)

        if perm_importance is None:
            print("Stopping: no model outperforms the dummy baseline.")
            break

        perm_importance.to_csv(res_dir / f"PERIM_{tag}.csv")

        if len(current_features) > 1:
            worst_feature = perm_importance["mean"].idxmin()
            print(f"Removing least important feature: {worst_feature}")
            current_features.remove(worst_feature)
        else:
            print("Reached the last remaining feature.")
            break

        step += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the input CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--regions", nargs="+", default=REGIONS, help="Regions to process")
    args = parser.parse_args()

    # encoding="utf-8-sig" strips a leading BOM (common in CSVs exported from Excel);
    # sep=None + engine="python" auto-detects the delimiter (comma, semicolon, tab, ...)
    EV_all = pd.read_csv(args.data, sep=None, engine="python", encoding="utf-8-sig")
    if "ID" not in EV_all.columns:
        raise ValueError(
            f"Column 'ID' not found in {args.data}. "
            f"Columns found: {list(EV_all.columns)}. "
            "Check the delimiter/encoding of the CSV, or adjust the column names in the script."
        )
    EV_all = EV_all.set_index("ID")

    for region in args.regions:
        run_region(EV_all, region, args.output_dir)


if __name__ == "__main__":
    main()