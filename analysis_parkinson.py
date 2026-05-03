from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif


SEED = 42
TEST_SIZE = 0.2
FEATURE_CANDIDATES = [10, 20, 30, 40, 60]
SCORING = {
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class StrategyResult:
    name: str
    summary: pd.DataFrame
    best_model_name: str
    best_model: Any
    best_predictions: np.ndarray
    best_probabilities: np.ndarray
    best_metrics: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare Parkinson voice classifiers on real and CTGAN synthetic data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("pd_speech_features.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where metrics, plots and reports will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Hold-out test proportion used for the final evaluation.",
    )
    parser.add_argument(
        "--ctgan-epochs",
        type=int,
        default=150,
        help="Number of CTGAN training epochs.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path, skiprows=1)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["gender"] = df["gender"].astype(int)
    y = df.pop("class").astype(int)
    return df, y


def rank_features(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    discrete_mask = np.array([column == "gender" for column in X_train.columns])
    mutual_info = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=discrete_mask,
        random_state=SEED,
    )

    forest = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    forest.fit(X_train, y_train)

    ranking = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mutual_info": mutual_info,
            "rf_importance": forest.feature_importances_,
        }
    )

    ranking["mutual_info_norm"] = ranking["mutual_info"] / max(ranking["mutual_info"].max(), 1e-12)
    ranking["rf_importance_norm"] = ranking["rf_importance"] / max(
        ranking["rf_importance"].max(),
        1e-12,
    )
    ranking["combined_score"] = 0.5 * (
        ranking["mutual_info_norm"] + ranking["rf_importance_norm"]
    )
    ranking = ranking.sort_values(
        by=["combined_score", "mutual_info", "rf_importance"],
        ascending=False,
    ).reset_index(drop=True)
    return ranking


def choose_feature_count(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ranking: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    selector_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    rows: list[dict[str, float]] = []
    for feature_count in FEATURE_CANDIDATES:
        chosen_features = ranking.head(feature_count)["feature"].tolist()
        scores = cross_val_score(
            selector_model,
            X_train[chosen_features],
            y_train,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        rows.append(
            {
                "feature_count": float(feature_count),
                "cv_balanced_accuracy_mean": float(scores.mean()),
                "cv_balanced_accuracy_std": float(scores.std(ddof=0)),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["cv_balanced_accuracy_mean", "cv_balanced_accuracy_std", "feature_count"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_feature_count = int(summary.iloc[0]["feature_count"])
    return best_feature_count, summary


def build_model_spaces() -> dict[str, dict[str, Any]]:
    return {
        "LogisticRegression": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "param_grid": {"model__C": [0.1, 1.0, 3.0, 10.0]},
        },
        "SVM": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            class_weight="balanced",
                            probability=True,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": [0.5, 1.0, 3.0],
                "model__gamma": ["scale", 0.01, 0.001],
                "model__kernel": ["rbf"],
            },
        },
        "KNN": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                "model__n_neighbors": [3, 5, 7, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
            "param_grid": {
                "n_estimators": [300, 500],
                "max_depth": [None, 12, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=SEED),
            "param_grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3],
            },
        },
    }


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def tune_and_evaluate_models(
    strategy_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> StrategyResult:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model_spaces = build_model_spaces()

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}

    for model_name, config in model_spaces.items():
        log(f"[{strategy_name}] Tuning {model_name}...")
        search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            scoring=SCORING,
            refit="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        fitted_models[model_name] = best_estimator

        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        probabilities[model_name] = y_prob
        predictions[model_name] = y_pred
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics_by_model[model_name] = metrics
        log(
            (
                f"[{strategy_name}] {model_name} done. "
                f"CV bal_acc={search.best_score_:.4f}, test bal_acc={metrics['balanced_accuracy']:.4f}"
            )
        )

        row = {
            "strategy": strategy_name,
            "model": model_name,
            "cv_balanced_accuracy_mean": float(
                search.cv_results_["mean_test_balanced_accuracy"][search.best_index_]
            ),
            "cv_balanced_accuracy_std": float(
                search.cv_results_["std_test_balanced_accuracy"][search.best_index_]
            ),
            "cv_f1_mean": float(search.cv_results_["mean_test_f1"][search.best_index_]),
            "cv_roc_auc_mean": float(
                search.cv_results_["mean_test_roc_auc"][search.best_index_]
            ),
            "test_accuracy": metrics["accuracy"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        by=["cv_balanced_accuracy_mean", "test_balanced_accuracy", "test_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    best_model_name = str(summary.iloc[0]["model"])
    return StrategyResult(
        name=strategy_name,
        summary=summary,
        best_model_name=best_model_name,
        best_model=fitted_models[best_model_name],
        best_predictions=predictions[best_model_name],
        best_probabilities=probabilities[best_model_name],
        best_metrics=metrics_by_model[best_model_name],
    )


def synthesize_training_data(
    X_train_selected: pd.DataFrame,
    y_train: pd.Series,
    epochs: int,
) -> pd.DataFrame:
    training_frame = X_train_selected.copy()
    training_frame["class"] = y_train.to_numpy()

    discrete_columns = ["class"]
    if "gender" in training_frame.columns:
        training_frame["gender"] = training_frame["gender"].astype(int)
        discrete_columns.append("gender")

    synthesizer = CTGAN(
        epochs=epochs,
        batch_size=100,
        pac=10,
        verbose=False,
        enable_gpu=False,
    )
    log(f"[synthetic] Training CTGAN for {epochs} epochs...")
    synthesizer.fit(training_frame, discrete_columns)
    log("[synthetic] CTGAN training finished. Sampling synthetic rows...")

    synthetic_parts: list[pd.DataFrame] = []
    for class_value, count in y_train.value_counts().sort_index().items():
        synthetic_parts.append(
            synthesizer.sample(
                int(count),
                condition_column="class",
                condition_value=int(class_value),
            )
        )

    synthetic = pd.concat(synthetic_parts, ignore_index=True)
    synthetic = synthetic.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    synthetic["class"] = synthetic["class"].round().astype(int)

    for column in X_train_selected.columns:
        if column == "gender":
            synthetic[column] = synthetic[column].round().clip(0, 1).astype(int)
            continue

        lower = X_train_selected[column].min()
        upper = X_train_selected[column].max()
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce").clip(lower, upper)

    synthetic = synthetic.dropna().reset_index(drop=True)
    return synthetic


def compute_synthetic_quality(
    real_selected_train: pd.DataFrame,
    synthetic_selected_train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    for column in real_selected_train.columns:
        real_values = real_selected_train[column].to_numpy()
        synthetic_values = synthetic_selected_train[column].to_numpy()
        ks_stat = float(ks_2samp(real_values, synthetic_values).statistic)
        wasserstein = float(wasserstein_distance(real_values, synthetic_values))
        scale = float(np.std(real_values, ddof=0))
        normalized_wasserstein = wasserstein / max(scale, 1e-12)
        rows.append(
            {
                "feature": column,
                "ks_stat": ks_stat,
                "wasserstein": wasserstein,
                "wasserstein_normalized": float(normalized_wasserstein),
            }
        )

    quality = pd.DataFrame(rows).sort_values(
        by=["ks_stat", "wasserstein_normalized"],
        ascending=False,
    ).reset_index(drop=True)

    real_corr = real_selected_train.corr(numeric_only=True).fillna(0.0).to_numpy()
    synthetic_corr = synthetic_selected_train.corr(numeric_only=True).fillna(0.0).to_numpy()
    correlation_mae = float(np.mean(np.abs(real_corr - synthetic_corr)))

    summary = {
        "mean_ks_stat": float(quality["ks_stat"].mean()),
        "max_ks_stat": float(quality["ks_stat"].max()),
        "mean_normalized_wasserstein": float(quality["wasserstein_normalized"].mean()),
        "correlation_mae": correlation_mae,
    }
    return quality, summary


def plot_feature_ranking(ranking: pd.DataFrame, figures_dir: Path) -> None:
    top = ranking.head(15).sort_values("combined_score", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["combined_score"], color="#2d6a4f")
    plt.xlabel("Combined feature relevance score")
    plt.ylabel("Feature")
    plt.title("Top 15 selected features")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_ranking.png", dpi=200)
    plt.close()


def plot_feature_count_search(feature_count_summary: pd.DataFrame, figures_dir: Path) -> None:
    ordered = feature_count_summary.sort_values("feature_count")
    plt.figure(figsize=(8, 5))
    plt.plot(
        ordered["feature_count"],
        ordered["cv_balanced_accuracy_mean"],
        marker="o",
        color="#1d3557",
    )
    plt.fill_between(
        ordered["feature_count"],
        ordered["cv_balanced_accuracy_mean"] - ordered["cv_balanced_accuracy_std"],
        ordered["cv_balanced_accuracy_mean"] + ordered["cv_balanced_accuracy_std"],
        color="#a8dadc",
        alpha=0.35,
    )
    plt.xlabel("Number of selected features")
    plt.ylabel("CV balanced accuracy")
    plt.title("Feature-count selection")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_count_selection.png", dpi=200)
    plt.close()


def plot_model_comparison(
    real_result: StrategyResult,
    synthetic_result: StrategyResult,
    figures_dir: Path,
) -> None:
    merged = real_result.summary[["model", "test_balanced_accuracy", "test_roc_auc"]].merge(
        synthetic_result.summary[["model", "test_balanced_accuracy", "test_roc_auc"]],
        on="model",
        suffixes=("_real", "_synthetic"),
    )
    merged = merged.sort_values("test_balanced_accuracy_real", ascending=True)
    positions = np.arange(len(merged))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].barh(
        positions - width / 2,
        merged["test_balanced_accuracy_real"],
        height=width,
        label="Real",
        color="#457b9d",
    )
    axes[0].barh(
        positions + width / 2,
        merged["test_balanced_accuracy_synthetic"],
        height=width,
        label="Synthetic",
        color="#e76f51",
    )
    axes[0].set_title("Balanced accuracy")
    axes[0].set_xlabel("Score")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(merged["model"])
    axes[0].legend(frameon=False)

    axes[1].barh(
        positions - width / 2,
        merged["test_roc_auc_real"],
        height=width,
        label="Real",
        color="#457b9d",
    )
    axes[1].barh(
        positions + width / 2,
        merged["test_roc_auc_synthetic"],
        height=width,
        label="Synthetic",
        color="#e76f51",
    )
    axes[1].set_title("ROC AUC")
    axes[1].set_xlabel("Score")

    fig.suptitle("Real vs synthetic training comparison")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=200)
    plt.close()


def plot_roc_curves(
    y_test: pd.Series,
    real_result: StrategyResult,
    synthetic_result: StrategyResult,
    figures_dir: Path,
) -> None:
    real_fpr, real_tpr, _ = roc_curve(y_test, real_result.best_probabilities)
    synthetic_fpr, synthetic_tpr, _ = roc_curve(y_test, synthetic_result.best_probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(
        real_fpr,
        real_tpr,
        label=(
            f"Real - {real_result.best_model_name} "
            f"(AUC={real_result.best_metrics['roc_auc']:.3f})"
        ),
        color="#1d3557",
        linewidth=2,
    )
    plt.plot(
        synthetic_fpr,
        synthetic_tpr,
        label=(
            f"Synthetic - {synthetic_result.best_model_name} "
            f"(AUC={synthetic_result.best_metrics['roc_auc']:.3f})"
        ),
        color="#e76f51",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on the real hold-out test set")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=200)
    plt.close()


def plot_confusion_matrices(
    y_test: pd.Series,
    real_result: StrategyResult,
    synthetic_result: StrategyResult,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        real_result.best_predictions,
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
    )
    axes[0].set_title(f"Real - {real_result.best_model_name}")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        synthetic_result.best_predictions,
        ax=axes[1],
        cmap="Oranges",
        colorbar=False,
    )
    axes[1].set_title(f"Synthetic - {synthetic_result.best_model_name}")

    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=200)
    plt.close()


def plot_synthetic_quality(quality: pd.DataFrame, figures_dir: Path) -> None:
    top = quality.head(12).sort_values("ks_stat", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["ks_stat"], color="#f4a261")
    plt.xlabel("KS statistic")
    plt.ylabel("Feature")
    plt.title("Largest real-vs-synthetic marginal drifts")
    plt.tight_layout()
    plt.savefig(figures_dir / "synthetic_quality.png", dpi=200)
    plt.close()


def write_report(
    output_dir: Path,
    selected_features: list[str],
    feature_count_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    real_result: StrategyResult,
    synthetic_result: StrategyResult,
    synthetic_quality: pd.DataFrame,
    synthetic_quality_summary: dict[str, float],
    dataset_summary: dict[str, Any],
    ctgan_epochs: int,
) -> None:
    real_table = real_result.summary.copy()
    synthetic_table = synthetic_result.summary.copy()
    for frame in (real_table, synthetic_table):
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_columns] = frame[numeric_columns].round(4)

    top_drift = synthetic_quality.head(5)[["feature", "ks_stat", "wasserstein_normalized"]].copy()
    top_drift[["ks_stat", "wasserstein_normalized"]] = top_drift[
        ["ks_stat", "wasserstein_normalized"]
    ].round(4)

    best_real = real_result.best_metrics
    best_synthetic = synthetic_result.best_metrics
    performance_gap = {
        metric: best_real[metric] - best_synthetic[metric]
        for metric in ["balanced_accuracy", "recall", "f1", "roc_auc"]
    }

    report = f"""# Parkinson voice classification analysis

## Dataset summary

- Samples: {dataset_summary['rows']}
- Predictive variables after removing `id`: {dataset_summary['features']}
- Target distribution: class 1 = {dataset_summary['class_counts'][1]}, class 0 = {dataset_summary['class_counts'][0]}
- Final hold-out split: {dataset_summary['train_rows']} train / {dataset_summary['test_rows']} test

## Methodology

1. A stratified train/test split was created before any feature selection or model fitting.
2. Feature relevance was ranked on the real training split using a combined score: mutual information + RandomForest importance.
3. The number of retained features was chosen with 5-fold CV and balanced accuracy.
4. Five classical models were tuned on the reduced feature space.
5. CTGAN was then fitted only on the selected features of the training split plus the target, and a purely synthetic training set with the same class counts was generated.
6. Both strategies were evaluated on the same real hold-out test split.

## Selected feature subset

- Selected feature count: {len(selected_features)}
- Selected features: {", ".join(selected_features)}

![Feature ranking](figures/feature_ranking.png)

![Feature count selection](figures/feature_count_selection.png)

## Real-data strategy

- Best CV model: {real_result.best_model_name}
- Hold-out balanced accuracy: {best_real['balanced_accuracy']:.4f}
- Hold-out recall: {best_real['recall']:.4f}
- Hold-out F1: {best_real['f1']:.4f}
- Hold-out ROC AUC: {best_real['roc_auc']:.4f}

{real_table.to_markdown(index=False)}

## CTGAN synthetic-data strategy

- CTGAN epochs: {ctgan_epochs}
- Best CV model: {synthetic_result.best_model_name}
- Hold-out balanced accuracy: {best_synthetic['balanced_accuracy']:.4f}
- Hold-out recall: {best_synthetic['recall']:.4f}
- Hold-out F1: {best_synthetic['f1']:.4f}
- Hold-out ROC AUC: {best_synthetic['roc_auc']:.4f}

{synthetic_table.to_markdown(index=False)}

## Comparison

- Balanced-accuracy gap (real - synthetic): {performance_gap['balanced_accuracy']:.4f}
- Recall gap (real - synthetic): {performance_gap['recall']:.4f}
- F1 gap (real - synthetic): {performance_gap['f1']:.4f}
- ROC AUC gap (real - synthetic): {performance_gap['roc_auc']:.4f}

The real-data strategy should be treated as the deployment reference if the synthetic strategy underperforms on recall or balanced accuracy. The synthetic model is still useful to discuss privacy-preserving augmentation, stress testing, and how much predictive signal CTGAN preserves after compression to the selected feature space.

![Model comparison](figures/model_comparison.png)

![ROC curves](figures/roc_curves.png)

![Confusion matrices](figures/confusion_matrices.png)

## Synthetic data quality

- Mean KS statistic across selected features: {synthetic_quality_summary['mean_ks_stat']:.4f}
- Max KS statistic: {synthetic_quality_summary['max_ks_stat']:.4f}
- Mean normalized Wasserstein distance: {synthetic_quality_summary['mean_normalized_wasserstein']:.4f}
- Correlation-matrix MAE: {synthetic_quality_summary['correlation_mae']:.4f}

The drift values below indicate which selected variables are hardest for CTGAN to reproduce faithfully. They are the best candidates to discuss when explaining why the synthetic classifier may lose sensitivity.

{top_drift.to_markdown(index=False)}

![Synthetic quality](figures/synthetic_quality.png)

## Conclusion

The complete comparison is now reproducible from a single script. If the real-data model wins by a clear margin, the interpretation is straightforward: synthetic data preserves part of the decision boundary, but not all of it. If the gap is small, you can argue that the selected voice descriptors retain enough structure for CTGAN to train a clinically useful proxy model.
"""

    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parent
    data_path = (base_dir / args.data).resolve() if not args.data.is_absolute() else args.data
    output_dir = (base_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log("Loading dataset...")
    X, y = load_dataset(data_path)
    log(f"Dataset loaded: {len(X)} rows, {X.shape[1]} features.")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=SEED,
    )

    log("Ranking features...")
    ranking = rank_features(X_train, y_train)
    best_feature_count, feature_count_summary = choose_feature_count(X_train, y_train, ranking)
    selected_features = ranking.head(best_feature_count)["feature"].tolist()
    log(f"Selected {best_feature_count} features.")

    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()

    log("Training classical models on real data...")
    real_result = tune_and_evaluate_models(
        "real",
        X_train_selected,
        y_train,
        X_test_selected,
        y_test,
    )

    log("Generating the synthetic training dataset...")
    synthetic_train = synthesize_training_data(
        X_train_selected,
        y_train,
        epochs=args.ctgan_epochs,
    )
    X_synthetic = synthetic_train[selected_features].copy()
    y_synthetic = synthetic_train["class"].astype(int)

    log("Training classical models on synthetic data...")
    synthetic_result = tune_and_evaluate_models(
        "synthetic",
        X_synthetic,
        y_synthetic,
        X_test_selected,
        y_test,
    )

    log("Computing synthetic-quality diagnostics and plots...")
    synthetic_quality, synthetic_quality_summary = compute_synthetic_quality(
        X_train_selected,
        X_synthetic,
    )

    plot_feature_ranking(ranking, figures_dir)
    plot_feature_count_search(feature_count_summary, figures_dir)
    plot_model_comparison(real_result, synthetic_result, figures_dir)
    plot_roc_curves(y_test, real_result, synthetic_result, figures_dir)
    plot_confusion_matrices(y_test, real_result, synthetic_result, figures_dir)
    plot_synthetic_quality(synthetic_quality, figures_dir)

    selected_feature_table = ranking.head(best_feature_count).copy()
    selected_feature_table.to_csv(output_dir / "selected_features.csv", index=False)
    feature_count_summary.to_csv(output_dir / "feature_count_search.csv", index=False)
    real_result.summary.to_csv(output_dir / "real_model_results.csv", index=False)
    synthetic_result.summary.to_csv(output_dir / "synthetic_model_results.csv", index=False)
    synthetic_quality.to_csv(output_dir / "synthetic_quality.csv", index=False)
    synthetic_train.to_csv(output_dir / "synthetic_training_data.csv", index=False)

    dataset_summary = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "class_counts": {0: int((y == 0).sum()), 1: int((y == 1).sum())},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    comparison_summary = {
        "dataset_summary": dataset_summary,
        "selected_feature_count": best_feature_count,
        "selected_features": selected_features,
        "real_best_model": real_result.best_model_name,
        "real_best_metrics": real_result.best_metrics,
        "synthetic_best_model": synthetic_result.best_model_name,
        "synthetic_best_metrics": synthetic_result.best_metrics,
        "synthetic_quality_summary": synthetic_quality_summary,
        "ctgan_epochs": args.ctgan_epochs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(comparison_summary, indent=2),
        encoding="utf-8",
    )

    write_report(
        output_dir=output_dir,
        selected_features=selected_features,
        feature_count_summary=feature_count_summary,
        ranking=ranking,
        real_result=real_result,
        synthetic_result=synthetic_result,
        synthetic_quality=synthetic_quality,
        synthetic_quality_summary=synthetic_quality_summary,
        dataset_summary=dataset_summary,
        ctgan_epochs=args.ctgan_epochs,
    )

    print(f"Analysis completed. Outputs written to: {output_dir}")
    print(f"Best real-data model: {real_result.best_model_name} -> {real_result.best_metrics}")
    print(f"Best synthetic-data model: {synthetic_result.best_model_name} -> {synthetic_result.best_metrics}")


if __name__ == "__main__":
    main()