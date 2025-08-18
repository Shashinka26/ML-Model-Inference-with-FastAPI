import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def load_from_builtin():
    ds = load_iris(as_frame=True)
    X = ds.data
    y = ds.target
    class_names = list(ds.target_names)
    feature_names = list(X.columns)
    return X, y, class_names, feature_names

def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Common Kaggle column names → normalize
    df = df.rename(columns={
        'SepalLengthCm': 'sepal_length',
        'SepalWidthCm': 'sepal_width',
        'PetalLengthCm': 'petal_length',
        'PetalWidthCm': 'petal_width',
        'Species': 'species',
        'Id': 'id'
    })
    # Drop id if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Expect these columns after rename
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    class_names = sorted(y.unique().tolist())
    feature_names = feature_cols
    return X, y, class_names, feature_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to Iris.csv (if not provided, uses sklearn built-in).")
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        X, y, class_names, feature_names = load_from_csv(args.csv)
    else:
        X, y, class_names, feature_names = load_from_builtin()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    bundle = {
        "model": pipe,
        "classes": class_names,
        "feature_names": feature_names,
        "test_accuracy": float(acc),
    }
    joblib.dump(bundle, "model.pkl")

    print("Model saved to model.pkl")
    print(f"Test accuracy: {acc:.4f}")
    try:
        print(classification_report(y_test, y_pred))
    except Exception:
        pass

if __name__ == "__main__":
    main()
