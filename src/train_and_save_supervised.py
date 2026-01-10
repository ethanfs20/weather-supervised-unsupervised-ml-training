import os
import json
import hashlib
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_TRAIN = "data/raw/rain_supervised_train.csv"
DATA_VAL   = "data/raw/rain_supervised_val.csv"

MODELS_DIR  = "models"
REPORTS_DIR = "reports"
RUNS_CSV    = os.path.join(REPORTS_DIR, "runs.csv")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = pd.read_csv(DATA_TRAIN)
    val_df = pd.read_csv(DATA_VAL)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].map({"NO": 0, "YES": 1})

    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"].map({"NO": 0, "YES": 1})

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "f1": f1_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
    }

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_path = os.path.join(MODELS_DIR, f"logreg_rain_{ts}.joblib")
    cols_path  = os.path.join(MODELS_DIR, f"feature_columns_{ts}.json")

    joblib.dump(model, model_path)

    with open(cols_path, "w") as f:
        json.dump(list(X_train.columns), f, indent=2)

    run_row = {
        "timestamp": ts,
        "model_type": "LogisticRegression",
        "model_path": model_path,
        "feature_columns_path": cols_path,
        "train_file": DATA_TRAIN,
        "val_file": DATA_VAL,
        "train_sha256": sha256_file(DATA_TRAIN),
        "val_sha256": sha256_file(DATA_VAL),
        **metrics,
    }

    if os.path.exists(RUNS_CSV):
        df = pd.read_csv(RUNS_CSV)
        df = pd.concat([df, pd.DataFrame([run_row])], ignore_index=True)
    else:
        df = pd.DataFrame([run_row])

    df.to_csv(RUNS_CSV, index=False)

    print("Model saved to:", model_path)
    print("Run logged to:", RUNS_CSV)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
