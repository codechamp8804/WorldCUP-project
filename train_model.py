from __future__ import annotations

import argparse
from pathlib import Path

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without joblib package
    # Older scikit-learn bundled joblib; use that if standalone joblib is unavailable
    from sklearn.externals import joblib  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore

from football_features import build_match_features, download_kaggle_dataset, load_results


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"


def train(data_path: Path, model_path: Path = MODEL_PATH) -> None:
    results = load_results(data_path)
    dataset = build_match_features(results)

    drop_columns = ["target", "date"]
    X = dataset.drop(columns=drop_columns)
    y = dataset["target"]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    split_index = int(len(dataset) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Rows: {len(dataset):,}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    print(f"Class order: {list(model.classes_)}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": list(model.classes_),
            "feature_columns": X.columns.tolist(),
            "teams": sorted(set(results["home_team"]).union(results["away_team"])),
            "tournaments": sorted(results["tournament"].dropna().unique()),
            "countries": sorted(results["country"].dropna().unique()),
            "results_path": str(data_path),
        },
        model_path,
    )
    print(f"\nSaved model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the football match outcome predictor.")
    parser.add_argument("--data", type=Path, default=Path("data/results.csv"), help="Path to results.csv")
    parser.add_argument("--download", action="store_true", help="Download the Kaggle dataset before training")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Where to save the trained model")
    args = parser.parse_args()

    data_path = download_kaggle_dataset(args.data.parent) if args.download else args.data
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Download results.csv from Kaggle into data/ "
            "or run with --download after configuring Kaggle credentials."
        )

    train(data_path, args.model)


if __name__ == "__main__":
    main()