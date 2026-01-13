import os
import json
import hashlib
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ---- DATA PATHS (lagged real dataset) ----
DATA_TRAIN = "data/processed/real_rain_train_lag.csv"
DATA_VAL   = "data/processed/real_rain_val_lag.csv"

MODELS_DIR  = "models"
REPORTS_DIR = "reports"
RUNS_CSV    = os.path.join(REPORTS_DIR, "runs.csv")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def main() -> None:
    ensure_dirs()

    # Load train/val
    train_df = pd.read_csv(DATA_TRAIN)
    val_df = pd.read_csv(DATA_VAL)

    if "label" not in train_df.columns or "label" not in val_df.columns:
        raise ValueError("Expected a 'label' column in both train and val CSVs")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].map({"NO": 0, "YES": 1}).astype(int)

    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"].map({"NO": 0, "YES": 1}).astype(int)

    # ---- Train (balanced classes) ----
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    pred_val = model.predict(X_val)

    acc = accuracy_score(y_val, pred_val)
    f1  = f1_score(y_val, pred_val)
    prec = precision_score(y_val, pred_val)
    rec  = recall_score(y_val, pred_val)
    cm = confusion_matrix(y_val, pred_val)  # [[TN FP],[FN TP]]

    # ---- Save artifacts ----
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"logreg_rain_{ts}.joblib")
    cols_path  = os.path.join(MODELS_DIR, f"feature_columns_{ts}.json")

    joblib.dump(model, model_path)

    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f, indent=2)

    # ---- Log run ----
    run_row = {
        "timestamp": ts,
        "model_type": "LogisticRegression",
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 2000,
        "train_file": DATA_TRAIN.replace("\\", "/"),
        "val_file": DATA_VAL.replace("\\", "/"),
        "train_sha256": sha256_file(DATA_TRAIN),
        "val_sha256": sha256_file(DATA_VAL),
        "model_path": model_path.replace("\\", "/"),
        "feature_columns_path": cols_path.replace("\\", "/"),
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    if os.path.exists(RUNS_CSV):
        runs = pd.read_csv(RUNS_CSV)
        runs = pd.concat([runs, pd.DataFrame([run_row])], ignore_index=True)
    else:
        runs = pd.DataFrame([run_row])

    runs.to_csv(RUNS_CSV, index=False)

    # ---- Print summary ----
    print("Saved model:", model_path)
    print("Saved feature columns:", cols_path)
    print("Logged run to:", RUNS_CSV)
    print(f"VAL accuracy={acc:.4f} f1={f1:.4f} precision={prec:.4f} recall={rec:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)


if __name__ == "__main__":
    main()
