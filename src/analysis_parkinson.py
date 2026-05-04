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
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
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
ROBUSTNESS_SEEDS = [13, 42, 97]
SCORING = {
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
STRATEGY_LABELS = {
    "real": "Datos reales",
    "synthetic": "CTGAN sintético",
    "mixed": "Real + sintético",
}
FEATURE_FAMILY_DESCRIPTIONS = {
    "TQWT / wavelet": (
        "Describe patrones multiescala en la voz; captura irregularidades temporales y espectrales "
        "que suelen aparecer cuando la fonación pierde estabilidad."
    ),
    "MFCC / energía / dinámicas": (
        "Resume la envolvente espectral y su evolución temporal; refleja cambios finos en timbre, "
        "articulación y control motor de la voz."
    ),
    "Jitter / shimmer": (
        "Mide microvariaciones ciclo a ciclo en frecuencia y amplitud; se asocia con inestabilidad "
        "de los pliegues vocales."
    ),
    "Intensidad": (
        "Caracteriza la energía y la proyección de la señal, útil para detectar alteraciones globales "
        "en la emisión vocal."
    ),
    "Otros descriptores vocales": (
        "Incluye rasgos acústicos secundarios que aportan contexto adicional, aunque no dominan la "
        "decisión del modelo."
    ),
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
    holdout_best_model_name: str
    holdout_best_metrics: dict[str, float]


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
        default=Path("data/pd_speech_features.csv"),
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
    parser.add_argument(
        "--robustness-seeds",
        nargs="*",
        type=int,
        default=ROBUSTNESS_SEEDS,
        help="Random seeds used in the lightweight robustness analysis.",
    )
    parser.add_argument(
        "--robustness-ctgan-epochs",
        type=int,
        default=75,
        help=(
            "CTGAN epochs used during the repeated-seed robustness analysis. "
            "This is lighter than the main experiment to keep runtime bounded."
        ),
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path, skiprows=1)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["gender"] = df["gender"].astype(int)
    y = df.pop("class").astype(int)
    return df, y


def rank_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int = SEED,
) -> pd.DataFrame:
    discrete_mask = np.array([column == "gender" for column in X_train.columns])
    mutual_info = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=discrete_mask,
        random_state=random_seed,
    )

    forest = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_seed,
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
    random_seed: int = SEED,
) -> tuple[int, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    selector_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_seed,
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


def build_model_spaces(random_seed: int = SEED) -> dict[str, dict[str, Any]]:
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
                            random_state=random_seed,
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
                            random_state=random_seed,
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
                random_state=random_seed,
            ),
            "param_grid": {
                "n_estimators": [300, 500],
                "max_depth": [None, 12, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=random_seed),
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def tune_and_evaluate_models(
    strategy_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_seed: int = SEED,
) -> StrategyResult:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    model_spaces = build_model_spaces(random_seed=random_seed)

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
            "test_sensitivity": metrics["sensitivity"],
            "test_specificity": metrics["specificity"],
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
    holdout_best_summary = summary.sort_values(
        by=["test_balanced_accuracy", "test_roc_auc", "cv_balanced_accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)
    holdout_best_model_name = str(holdout_best_summary.iloc[0]["model"])
    return StrategyResult(
        name=strategy_name,
        summary=summary,
        best_model_name=best_model_name,
        best_model=fitted_models[best_model_name],
        best_predictions=predictions[best_model_name],
        best_probabilities=probabilities[best_model_name],
        best_metrics=metrics_by_model[best_model_name],
        holdout_best_model_name=holdout_best_model_name,
        holdout_best_metrics=metrics_by_model[holdout_best_model_name],
    )


def synthesize_training_data(
    X_train_selected: pd.DataFrame,
    y_train: pd.Series,
    epochs: int,
    random_seed: int = SEED,
) -> pd.DataFrame:
    set_seed(random_seed)
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


def classify_feature_family(feature_name: str) -> str:
    lower = feature_name.lower()
    if lower.startswith("tqwt_"):
        return "TQWT / wavelet"
    if "jitter" in lower or "shimmer" in lower:
        return "Jitter / shimmer"
    if "mfcc" in lower or "delta" in lower or "log_energy" in lower:
        return "MFCC / energía / dinámicas"
    if "intensity" in lower:
        return "Intensidad"
    return "Otros descriptores vocales"


def summarize_feature_families(selected_feature_table: pd.DataFrame) -> pd.DataFrame:
    enriched = selected_feature_table.copy()
    enriched["family"] = enriched["feature"].map(classify_feature_family)
    summary = (
        enriched.groupby("family", as_index=False)
        .agg(
            count=("feature", "count"),
            mean_combined_score=("combined_score", "mean"),
            features=("feature", lambda values: ", ".join(values)),
        )
        .sort_values(by=["count", "mean_combined_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["share"] = summary["count"] / max(int(enriched.shape[0]), 1)
    summary["description"] = summary["family"].map(FEATURE_FAMILY_DESCRIPTIONS)
    return summary[
        ["family", "count", "share", "mean_combined_score", "description", "features"]
    ]


def build_estimator_from_selection(
    model_name: str,
    best_params_json: str,
    random_seed: int,
) -> Any:
    estimator = clone(build_model_spaces(random_seed=random_seed)[model_name]["estimator"])
    estimator.set_params(**json.loads(best_params_json))
    return estimator


def evaluate_fixed_model(
    model_name: str,
    best_params_json: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_seed: int,
) -> dict[str, float]:
    estimator = build_estimator_from_selection(model_name, best_params_json, random_seed)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    return evaluate_predictions(y_test, predictions, probabilities)


def run_robustness_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    selected_feature_count: int,
    strategy_specs: dict[str, dict[str, str]],
    test_size: float,
    ctgan_epochs: int,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        log(f"[robustness] Evaluating seed {seed}...")
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )

        ranking = rank_features(X_train, y_train, random_seed=seed)
        selected_features = ranking.head(selected_feature_count)["feature"].tolist()
        X_train_selected = X_train[selected_features].copy()
        X_test_selected = X_test[selected_features].copy()

        real_metrics = evaluate_fixed_model(
            strategy_specs["real"]["model"],
            strategy_specs["real"]["best_params"],
            X_train_selected,
            y_train,
            X_test_selected,
            y_test,
            random_seed=seed,
        )

        synthetic_train = synthesize_training_data(
            X_train_selected,
            y_train,
            epochs=ctgan_epochs,
            random_seed=seed,
        )
        X_synthetic = synthetic_train[selected_features].copy()
        y_synthetic = synthetic_train["class"].astype(int)

        synthetic_metrics = evaluate_fixed_model(
            strategy_specs["synthetic"]["model"],
            strategy_specs["synthetic"]["best_params"],
            X_synthetic,
            y_synthetic,
            X_test_selected,
            y_test,
            random_seed=seed,
        )

        X_mixed = pd.concat([X_train_selected, X_synthetic], ignore_index=True)
        y_mixed = pd.concat([y_train.reset_index(drop=True), y_synthetic], ignore_index=True)
        mixed_metrics = evaluate_fixed_model(
            strategy_specs["mixed"]["model"],
            strategy_specs["mixed"]["best_params"],
            X_mixed,
            y_mixed,
            X_test_selected,
            y_test,
            random_seed=seed,
        )

        for strategy_name, metrics in {
            "real": real_metrics,
            "synthetic": synthetic_metrics,
            "mixed": mixed_metrics,
        }.items():
            row = {
                "seed": seed,
                "strategy": strategy_name,
                "model": strategy_specs[strategy_name]["model"],
            }
            row.update(metrics)
            rows.append(row)

    runs = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    ]
    for strategy_name, group in runs.groupby("strategy"):
        row = {
            "strategy": strategy_name,
            "model": str(group["model"].iloc[0]),
        }
        for metric_name in metric_names:
            row[f"{metric_name}_mean"] = float(group[metric_name].mean())
            row[f"{metric_name}_std"] = float(group[metric_name].std(ddof=0))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        by=["balanced_accuracy_mean", "roc_auc_mean"],
        ascending=False,
    ).reset_index(drop=True)
    return runs, summary


def round_metric_dict(metrics: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: round(float(value), digits) for key, value in metrics.items()}


def dataframe_records(frame: pd.DataFrame, digits: int = 4) -> list[dict[str, Any]]:
    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(digits)
    return json.loads(rounded.to_json(orient="records"))


def make_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir)).replace("\\", "/")
    except ValueError:
        return path.name


def serialize_strategy_result(result: StrategyResult) -> dict[str, Any]:
    return {
        "selected_model_by_cv": result.best_model_name,
        "selected_metrics": round_metric_dict(result.best_metrics),
        "holdout_best_model": result.holdout_best_model_name,
        "holdout_best_metrics": round_metric_dict(result.holdout_best_metrics),
        "summary_table": dataframe_records(result.summary),
    }


def write_app_data(
    app_dir: Path,
    output_dir: Path,
    dataset_summary: dict[str, Any],
    selected_feature_table: pd.DataFrame,
    feature_family_summary: pd.DataFrame,
    strategy_results: dict[str, StrategyResult],
    synthetic_quality: pd.DataFrame,
    synthetic_quality_summary: dict[str, float],
    robustness_summary: pd.DataFrame,
    robustness_seeds: list[int],
    robustness_ctgan_epochs: int,
) -> None:
    top_features = selected_feature_table.head(12).copy()
    top_features["family"] = top_features["feature"].map(classify_feature_family)
    top_drift = synthetic_quality.head(6)[
        ["feature", "ks_stat", "wasserstein_normalized"]
    ].copy()
    top_drift["family"] = top_drift["feature"].map(classify_feature_family)

    figures_dir = output_dir / "figures"
    app_data = {
        "project": {
            "title": "ParkinVoice: Parkinson detection from voice descriptors",
            "subtitle": (
                "Comparative analysis between classical models trained on real data, purely synthetic "
                "CTGAN data, and a mixed scenario."
            ),
        },
        "dataset": dataset_summary,
        "methodology": [
            "Estratificar train/test antes de cualquier selección de variables.",
            "Rankear variables con información mutua e importancia de RandomForest.",
            "Seleccionar un subconjunto compacto de rasgos vocales.",
            "Comparar entrenamiento con datos reales, sintéticos y mixtos sobre el mismo test real.",
            "Analizar deriva de CTGAN y estabilidad de los resultados con varias semillas.",
        ],
        "figures": {
            "featureRanking": make_relative_path(figures_dir / "feature_ranking.png", app_dir),
            "featureCount": make_relative_path(figures_dir / "feature_count_selection.png", app_dir),
            "featureFamilies": make_relative_path(figures_dir / "feature_family_summary.png", app_dir),
            "modelComparison": make_relative_path(figures_dir / "model_comparison.png", app_dir),
            "rocCurves": make_relative_path(figures_dir / "roc_curves.png", app_dir),
            "confusionMatrices": make_relative_path(figures_dir / "confusion_matrices.png", app_dir),
            "syntheticQuality": make_relative_path(figures_dir / "synthetic_quality.png", app_dir),
            "robustness": make_relative_path(figures_dir / "robustness_summary.png", app_dir),
        },
        "selectedFeatureCount": int(selected_feature_table.shape[0]),
        "topFeatures": dataframe_records(top_features[["feature", "combined_score", "family"]]),
        "featureFamilies": dataframe_records(feature_family_summary),
        "strategies": {
            key: serialize_strategy_result(value) for key, value in strategy_results.items()
        },
        "syntheticQuality": {
            "summary": round_metric_dict(synthetic_quality_summary),
            "topDrift": dataframe_records(top_drift),
        },
        "robustness": {
            "seeds": robustness_seeds,
            "ctgan_epochs": robustness_ctgan_epochs,
            "summary": dataframe_records(robustness_summary),
        },
        "conclusions": [
            "Los datos reales siguen siendo la referencia de despliegue: maximizan balanced accuracy y ROC AUC.",
            "CTGAN conserva parte de la señal discriminativa, pero no reemplaza al entrenamiento real en este dataset.",
            "La familia TQWT domina la selección, lo que sugiere que la información multiescala es crítica para detectar Parkinson en la voz.",
            "La deriva sintética más fuerte aparece en rasgos de jitter y en varias descomposiciones TQWT, coherente con la caída de rendimiento.",
            "El escenario mixto responde a la pregunta práctica importante: el sintético puro empeora, pero puede seguir siendo útil como complemento experimental.",
        ],
    }

    app_dir.mkdir(parents=True, exist_ok=True)
    app_data_path = app_dir / "app-data.js"
    app_data_path.write_text(
        "window.APP_DATA = " + json.dumps(app_data, indent=2) + ";\n",
        encoding="utf-8",
    )


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


def plot_feature_family_summary(feature_family_summary: pd.DataFrame, figures_dir: Path) -> None:
    ordered = feature_family_summary.sort_values("count", ascending=True)
    plt.figure(figsize=(9, 5.5))
    plt.barh(ordered["family"], ordered["count"], color="#264653")
    plt.xlabel("Number of selected features")
    plt.ylabel("Feature family")
    plt.title("Selected feature families")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_family_summary.png", dpi=200)
    plt.close()


def plot_model_comparison(strategy_results: dict[str, StrategyResult], figures_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for strategy_name, result in strategy_results.items():
        table = result.summary[["model", "test_balanced_accuracy", "test_roc_auc"]].copy()
        table["strategy"] = strategy_name
        rows.extend(table.to_dict(orient="records"))

    combined = pd.DataFrame(rows)
    models = combined["model"].drop_duplicates().tolist()
    strategies = [name for name in ["real", "synthetic", "mixed"] if name in combined["strategy"].unique()]
    x = np.arange(len(models))
    width = 0.24
    colors = {"real": "#457b9d", "synthetic": "#e76f51", "mixed": "#2a9d8f"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for axis, metric, title in [
        (axes[0], "test_balanced_accuracy", "Balanced accuracy"),
        (axes[1], "test_roc_auc", "ROC AUC"),
    ]:
        for index, strategy_name in enumerate(strategies):
            offsets = x + (index - (len(strategies) - 1) / 2) * width
            values = [
                float(
                    combined.loc[
                        (combined["model"] == model) & (combined["strategy"] == strategy_name),
                        metric,
                    ].iloc[0]
                )
                for model in models
            ]
            axis.bar(
                offsets,
                values,
                width=width,
                label=STRATEGY_LABELS[strategy_name],
                color=colors[strategy_name],
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(models, rotation=25, ha="right")
        axis.set_ylim(0.45, 1.0)
        axis.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False)
    fig.suptitle("Model family performance across training strategies")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=200)
    plt.close()


def plot_roc_curves(
    y_test: pd.Series,
    strategy_results: dict[str, StrategyResult],
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    colors = {"real": "#1d3557", "synthetic": "#e76f51", "mixed": "#2a9d8f"}
    for strategy_name in ["real", "synthetic", "mixed"]:
        if strategy_name not in strategy_results:
            continue
        result = strategy_results[strategy_name]
        fpr, tpr, _ = roc_curve(y_test, result.best_probabilities)
        plt.plot(
            fpr,
            tpr,
            label=(
                f"{STRATEGY_LABELS[strategy_name]} - {result.best_model_name} "
                f"(AUC={result.best_metrics['roc_auc']:.3f})"
            ),
            color=colors[strategy_name],
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
    strategy_results: dict[str, StrategyResult],
    figures_dir: Path,
) -> None:
    strategy_names = [name for name in ["real", "synthetic", "mixed"] if name in strategy_results]
    fig, axes = plt.subplots(1, len(strategy_names), figsize=(5 * len(strategy_names), 4.5))
    axes = np.atleast_1d(axes)
    cmap_map = {"real": "Blues", "synthetic": "Oranges", "mixed": "Greens"}

    for axis, strategy_name in zip(axes, strategy_names):
        result = strategy_results[strategy_name]
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            result.best_predictions,
            ax=axis,
            cmap=cmap_map[strategy_name],
            colorbar=False,
        )
        axis.set_title(f"{STRATEGY_LABELS[strategy_name]} - {result.best_model_name}")

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


def plot_robustness_summary(robustness_summary: pd.DataFrame, figures_dir: Path) -> None:
    ordered = robustness_summary.copy()
    ordered["label"] = ordered["strategy"].map(STRATEGY_LABELS)
    colors = {"real": "#457b9d", "synthetic": "#e76f51", "mixed": "#2a9d8f"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, metric, title in [
        (axes[0], "balanced_accuracy", "Balanced accuracy stability"),
        (axes[1], "roc_auc", "ROC AUC stability"),
    ]:
        axis.bar(
            ordered["label"],
            ordered[f"{metric}_mean"],
            yerr=ordered[f"{metric}_std"],
            color=[colors[name] for name in ordered["strategy"]],
            capsize=6,
        )
        axis.set_title(title)
        axis.set_ylabel("Mean ± std")
        axis.set_ylim(0.45, 1.0)
        axis.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(figures_dir / "robustness_summary.png", dpi=200)
    plt.close()


def write_report(
    output_dir: Path,
    selected_features: list[str],
    feature_count_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    feature_family_summary: pd.DataFrame,
    real_result: StrategyResult,
    synthetic_result: StrategyResult,
    mixed_result: StrategyResult,
    synthetic_quality: pd.DataFrame,
    synthetic_quality_summary: dict[str, float],
    robustness_summary: pd.DataFrame,
    dataset_summary: dict[str, Any],
    ctgan_epochs: int,
    robustness_ctgan_epochs: int,
    robustness_seeds: list[int],
) -> None:
    real_table = real_result.summary.copy()
    synthetic_table = synthetic_result.summary.copy()
    mixed_table = mixed_result.summary.copy()
    robustness_table = robustness_summary.copy()
    family_table = feature_family_summary.copy()
    for frame in (real_table, synthetic_table, mixed_table, robustness_table, family_table):
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_columns] = frame[numeric_columns].round(4)

    top_drift = synthetic_quality.head(5)[["feature", "ks_stat", "wasserstein_normalized"]].copy()
    top_drift[["ks_stat", "wasserstein_normalized"]] = top_drift[
        ["ks_stat", "wasserstein_normalized"]
    ].round(4)

    best_real = real_result.best_metrics
    best_synthetic = synthetic_result.best_metrics
    best_mixed = mixed_result.best_metrics
    performance_gap = {
        metric: best_real[metric] - best_synthetic[metric]
        for metric in ["balanced_accuracy", "sensitivity", "f1", "roc_auc"]
    }
    dominant_family = str(family_table.iloc[0]["family"])

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
6. A third mixed strategy was added by concatenating real and synthetic training rows.
7. A lightweight robustness analysis repeated the selected strategies across seeds {robustness_seeds}.

## Selected feature subset

- Selected feature count: {len(selected_features)}
- Selected features: {", ".join(selected_features)}
- Dominant family: {dominant_family}

### Family-level interpretation

The selected variables are not random column names: they form coherent acoustic families. TQWT descriptors dominate the subset, which suggests that multi-scale irregularities in the voice carry strong information about Parkinson-related degradation. MFCC and dynamic-energy features contribute complementary spectral and temporal structure, while jitter/shimmer variables capture cycle-to-cycle instability in phonation. Intensity appears less often, but it still adds a global energetic cue.

{family_table.to_markdown(index=False)}

![Feature ranking](figures/feature_ranking.png)

![Feature count selection](figures/feature_count_selection.png)

![Feature families](figures/feature_family_summary.png)

## Real-data strategy

- CV-selected model: {real_result.best_model_name}
- Best hold-out model by balanced accuracy: {real_result.holdout_best_model_name}
- Hold-out balanced accuracy: {best_real['balanced_accuracy']:.4f}
- Hold-out sensitivity: {best_real['sensitivity']:.4f}
- Hold-out specificity: {best_real['specificity']:.4f}
- Hold-out F1: {best_real['f1']:.4f}
- Hold-out ROC AUC: {best_real['roc_auc']:.4f}

{real_table.to_markdown(index=False)}

## CTGAN synthetic-data strategy

- CTGAN epochs: {ctgan_epochs}
- CV-selected model: {synthetic_result.best_model_name}
- Best hold-out model by balanced accuracy: {synthetic_result.holdout_best_model_name}
- Hold-out balanced accuracy: {best_synthetic['balanced_accuracy']:.4f}
- Hold-out sensitivity: {best_synthetic['sensitivity']:.4f}
- Hold-out specificity: {best_synthetic['specificity']:.4f}
- Hold-out F1: {best_synthetic['f1']:.4f}
- Hold-out ROC AUC: {best_synthetic['roc_auc']:.4f}

{synthetic_table.to_markdown(index=False)}

## Mixed real + CTGAN strategy

- CV-selected model: {mixed_result.best_model_name}
- Best hold-out model by balanced accuracy: {mixed_result.holdout_best_model_name}
- Hold-out balanced accuracy: {best_mixed['balanced_accuracy']:.4f}
- Hold-out sensitivity: {best_mixed['sensitivity']:.4f}
- Hold-out specificity: {best_mixed['specificity']:.4f}
- Hold-out F1: {best_mixed['f1']:.4f}
- Hold-out ROC AUC: {best_mixed['roc_auc']:.4f}

{mixed_table.to_markdown(index=False)}

## Comparison

- Balanced-accuracy gap (real - synthetic): {performance_gap['balanced_accuracy']:.4f}
- Sensitivity gap (real - synthetic): {performance_gap['sensitivity']:.4f}
- F1 gap (real - synthetic): {performance_gap['f1']:.4f}
- ROC AUC gap (real - synthetic): {performance_gap['roc_auc']:.4f}

The real-data strategy should still be treated as the deployment reference. The synthetic strategy is useful precisely because it underperforms: it shows that CTGAN preserves part of the signal but not enough to replace real clinical data. The mixed strategy answers the practical question that matters for the presentation: even when synthetic-only training is weaker, synthetic rows may still serve as an auxiliary experimental resource.

![Model comparison](figures/model_comparison.png)

![ROC curves](figures/roc_curves.png)

![Confusion matrices](figures/confusion_matrices.png)

## Robustness across seeds

The table below repeats the CV-selected model family for each strategy over seeds {robustness_seeds}. This is not a full nested re-tuning study; it is a stability check designed to estimate how much the conclusions move when the train/test split and CTGAN fitting are perturbed. For runtime reasons, CTGAN was retrained with {robustness_ctgan_epochs} epochs in this robustness block.

{robustness_table.to_markdown(index=False)}

![Robustness summary](figures/robustness_summary.png)

## Synthetic data quality

- Mean KS statistic across selected features: {synthetic_quality_summary['mean_ks_stat']:.4f}
- Max KS statistic: {synthetic_quality_summary['max_ks_stat']:.4f}
- Mean normalized Wasserstein distance: {synthetic_quality_summary['mean_normalized_wasserstein']:.4f}
- Correlation-matrix MAE: {synthetic_quality_summary['correlation_mae']:.4f}

The drift values below indicate which selected variables are hardest for CTGAN to reproduce faithfully. They are the best candidates to discuss when explaining why the synthetic classifier may lose sensitivity.

{top_drift.to_markdown(index=False)}

![Synthetic quality](figures/synthetic_quality.png)

## Conclusion

The complete comparison is now reproducible from a single script. The mature interpretation is not that CTGAN "failed", but that it preserved only part of the clinically useful signal. In this dataset, the combination of limited sample size and very high-dimensional multiscale voice descriptors makes high-fidelity synthetic generation difficult. That weakness strengthens the oral defence: it shows that you are not overselling synthetic data, and that you understand where the method helps and where it does not replace real evidence.
"""

    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    app_dir = project_dir / "docs"
    data_path = (project_dir / args.data).resolve() if not args.data.is_absolute() else args.data
    output_dir = (project_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
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
    ranking = rank_features(X_train, y_train, random_seed=SEED)
    best_feature_count, feature_count_summary = choose_feature_count(
        X_train,
        y_train,
        ranking,
        random_seed=SEED,
    )
    selected_features = ranking.head(best_feature_count)["feature"].tolist()
    log(f"Selected {best_feature_count} features.")

    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    selected_feature_table = ranking.head(best_feature_count).copy()
    feature_family_summary = summarize_feature_families(selected_feature_table)

    log("Training classical models on real data...")
    real_result = tune_and_evaluate_models(
        "real",
        X_train_selected,
        y_train,
        X_test_selected,
        y_test,
        random_seed=SEED,
    )

    log("Generating the synthetic training dataset...")
    synthetic_train = synthesize_training_data(
        X_train_selected,
        y_train,
        epochs=args.ctgan_epochs,
        random_seed=SEED,
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
        random_seed=SEED,
    )

    X_mixed = pd.concat([X_train_selected, X_synthetic], ignore_index=True)
    y_mixed = pd.concat([y_train.reset_index(drop=True), y_synthetic], ignore_index=True)
    log("Training classical models on mixed data...")
    mixed_result = tune_and_evaluate_models(
        "mixed",
        X_mixed,
        y_mixed,
        X_test_selected,
        y_test,
        random_seed=SEED,
    )

    log("Computing synthetic-quality diagnostics and plots...")
    synthetic_quality, synthetic_quality_summary = compute_synthetic_quality(
        X_train_selected,
        X_synthetic,
    )

    strategy_specs = {
        result.name: {
            "model": str(result.summary.iloc[0]["model"]),
            "best_params": str(result.summary.iloc[0]["best_params"]),
        }
        for result in [real_result, synthetic_result, mixed_result]
    }
    log("Running lightweight robustness analysis across multiple seeds...")
    robustness_runs, robustness_summary = run_robustness_analysis(
        X,
        y,
        selected_feature_count=best_feature_count,
        strategy_specs=strategy_specs,
        test_size=args.test_size,
        ctgan_epochs=args.robustness_ctgan_epochs,
        seeds=args.robustness_seeds,
    )

    strategy_results = {
        "real": real_result,
        "synthetic": synthetic_result,
        "mixed": mixed_result,
    }

    plot_feature_ranking(ranking, figures_dir)
    plot_feature_count_search(feature_count_summary, figures_dir)
    plot_feature_family_summary(feature_family_summary, figures_dir)
    plot_model_comparison(strategy_results, figures_dir)
    plot_roc_curves(y_test, strategy_results, figures_dir)
    plot_confusion_matrices(y_test, strategy_results, figures_dir)
    plot_synthetic_quality(synthetic_quality, figures_dir)
    plot_robustness_summary(robustness_summary, figures_dir)

    selected_feature_table.to_csv(output_dir / "selected_features.csv", index=False)
    feature_family_summary.to_csv(output_dir / "feature_family_summary.csv", index=False)
    feature_count_summary.to_csv(output_dir / "feature_count_search.csv", index=False)
    real_result.summary.to_csv(output_dir / "real_model_results.csv", index=False)
    synthetic_result.summary.to_csv(output_dir / "synthetic_model_results.csv", index=False)
    mixed_result.summary.to_csv(output_dir / "mixed_model_results.csv", index=False)
    synthetic_quality.to_csv(output_dir / "synthetic_quality.csv", index=False)
    synthetic_train.to_csv(output_dir / "synthetic_training_data.csv", index=False)
    robustness_runs.to_csv(output_dir / "robustness_runs.csv", index=False)
    robustness_summary.to_csv(output_dir / "robustness_summary.csv", index=False)

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
        "feature_family_summary": dataframe_records(feature_family_summary),
        "strategies": {
            key: serialize_strategy_result(value) for key, value in strategy_results.items()
        },
        "synthetic_quality_summary": synthetic_quality_summary,
        "ctgan_epochs": args.ctgan_epochs,
        "robustness": {
            "seeds": args.robustness_seeds,
            "ctgan_epochs": args.robustness_ctgan_epochs,
            "summary": dataframe_records(robustness_summary),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(comparison_summary, indent=2),
        encoding="utf-8",
    )

    write_app_data(
        app_dir=app_dir,
        output_dir=output_dir,
        dataset_summary=dataset_summary,
        selected_feature_table=selected_feature_table,
        feature_family_summary=feature_family_summary,
        strategy_results=strategy_results,
        synthetic_quality=synthetic_quality,
        synthetic_quality_summary=synthetic_quality_summary,
        robustness_summary=robustness_summary,
        robustness_seeds=args.robustness_seeds,
        robustness_ctgan_epochs=args.robustness_ctgan_epochs,
    )

    write_report(
        output_dir=output_dir,
        selected_features=selected_features,
        feature_count_summary=feature_count_summary,
        ranking=ranking,
        feature_family_summary=feature_family_summary,
        real_result=real_result,
        synthetic_result=synthetic_result,
        mixed_result=mixed_result,
        synthetic_quality=synthetic_quality,
        synthetic_quality_summary=synthetic_quality_summary,
        robustness_summary=robustness_summary,
        dataset_summary=dataset_summary,
        ctgan_epochs=args.ctgan_epochs,
        robustness_ctgan_epochs=args.robustness_ctgan_epochs,
        robustness_seeds=args.robustness_seeds,
    )

    print(f"Analysis completed. Outputs written to: {output_dir}")
    print(f"Best real-data model: {real_result.best_model_name} -> {real_result.best_metrics}")
    print(f"Best synthetic-data model: {synthetic_result.best_model_name} -> {synthetic_result.best_metrics}")
    print(f"Best mixed-data model: {mixed_result.best_model_name} -> {mixed_result.best_metrics}")


if __name__ == "__main__":
    main()