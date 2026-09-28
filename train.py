"""Train and evaluate the email threat classification baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.email_threat.dataset import OUTPUT_PATH
from src.email_threat.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "email_threat_model.joblib"
REPORTS_DIR = ROOT / "reports"


def train_model(
    dataset_path: Path = OUTPUT_PATH,
    model_path: Path = MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run `python -m src.email_threat.dataset` first."
        )

    data = pd.read_csv(dataset_path)
    missing = set(FEATURE_NAMES + ["label"]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if data.empty or data["label"].nunique() != 2:
        raise ValueError("Dataset must contain records from both classes.")

    reports_dir.mkdir(parents=True, exist_ok=True)
    _save_eda(data, reports_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_NAMES],
        data["label"].astype(int),
        test_size=test_size,
        random_state=seed,
        stratify=data["label"].astype(int),
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics: dict[str, object] = {
        "dataset": "Apache SpamAssassin Public Corpus (2003-02-28)",
        "records": int(len(data)),
        "train_records": int(len(x_train)),
        "test_records": int(len(x_test)),
        "test_size": test_size,
        "seed": seed,
        "features": FEATURE_NAMES,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": [
            [int(value) for value in row]
            for row in confusion_matrix(y_test, predictions, labels=[0, 1])
        ],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_path)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=[0, 1], display_labels=["Ham", "Spam"], cmap="Blues"
    )
    plt.title("Email threat classification — confusion matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Email threat classification — ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=160)
    plt.close()

    coefficients = model.named_steps["classifier"].coef_[0]
    importance = pd.Series(coefficients, index=FEATURE_NAMES).sort_values()
    colors = ["#d95f02" if value < 0 else "#1b9e77" for value in importance]
    plt.figure(figsize=(8, 6))
    importance.plot.barh(color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardized Logistic Regression coefficient")
    plt.title("Feature associations (not causal effects)")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_coefficients.png", dpi=160)
    plt.close()
    return metrics


def _save_eda(data: pd.DataFrame, reports_dir: Path) -> None:
    labels = data["label"].astype(int)
    counts = labels.value_counts().sort_index()
    numeric = data[FEATURE_NAMES]
    quartiles = numeric.quantile([0.25, 0.75])
    iqr = quartiles.loc[0.75] - quartiles.loc[0.25]
    outlier_counts = (
        (numeric.lt(quartiles.loc[0.25] - 1.5 * iqr))
        | (numeric.gt(quartiles.loc[0.75] + 1.5 * iqr))
    ).sum()
    profile: dict[str, object] = {
        "records": int(len(data)),
        "class_counts": {"ham": int(counts.get(0, 0)), "spam": int(counts.get(1, 0))},
        "class_proportions": {
            "ham": float(counts.get(0, 0) / len(data)),
            "spam": float(counts.get(1, 0) / len(data)),
        },
        "missing_values": {
            str(name): int(value) for name, value in data[["label", *FEATURE_NAMES]].isna().sum().items()
        },
        "duplicate_feature_rows": int(numeric.duplicated().sum()),
        "numeric_summary": {
            name: {
                "mean": float(numeric[name].mean()),
                "median": float(numeric[name].median()),
                "min": float(numeric[name].min()),
                "max": float(numeric[name].max()),
                "iqr_outlier_count": int(outlier_counts[name]),
            }
            for name in FEATURE_NAMES
        },
        "feature_medians_by_class": {
            str(label): {name: float(value) for name, value in row.items()}
            for label, row in data.groupby("label")[FEATURE_NAMES].median().iterrows()
        },
    }
    (reports_dir / "data_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )

    plt.figure(figsize=(5, 4))
    plt.bar(["Ham", "Spam"], [counts.get(0, 0), counts.get(1, 0)], color=["#4c78a8", "#e45756"])
    plt.ylabel("Email records")
    plt.title("Dataset class distribution")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution.png", dpi=160)
    plt.close()

    figure, axes = plt.subplots(5, 3, figsize=(13, 16))
    for axis, feature in zip(axes.flat, FEATURE_NAMES):
        grouped = [numeric.loc[labels == label, feature].dropna() for label in (0, 1)]
        axis.boxplot(grouped, showfliers=False)
        axis.set_xticks([1, 2], labels=["Ham", "Spam"])
        axis.set_title(feature.replace("_", " "), fontsize=9)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle("Feature distributions by class (outliers hidden for readability)")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(reports_dir / "feature_distributions.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    results = train_model(test_size=arguments.test_size, seed=arguments.seed)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
