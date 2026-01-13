import os
import json
import glob

import joblib
import pandas as pd


MODELS_DIR = "models"
VAL_CSV = "data/processed/real_rain_val_lag.csv"



def newest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    return max(files, key=os.path.getmtime)


def main():
    # Find most recent model + its feature columns file
    model_path = newest_file(os.path.join(MODELS_DIR, "logreg_rain_*.joblib"))
    cols_path = newest_file(os.path.join(MODELS_DIR, "feature_columns_*.json"))

    print("Loading model:", model_path)
    print("Loading columns:", cols_path)

    model = joblib.load(model_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        cols = json.load(f)

    # Load some data to predict on (validation set)
    df = pd.read_csv(VAL_CSV)

    # Separate X, keep label for comparison
    y_true = df["label"].copy()
    X = df.drop(columns=["label"])

    # Enforce same column order used during training
    missing = [c for c in cols if c not in X.columns]
    extra = [c for c in X.columns if c not in cols]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    if extra:
        # Not fatal, but we drop extras to be safe
        X = X[cols]
    else:
        X = X[cols]

        # Predict probability of YES (rain tomorrow)
    p_yes = model.predict_proba(X)[:, 1]
    pred_yes = (p_yes >= 0.5).astype(int)

    # Build output
    out = df.drop(columns=["label"]).copy()
    out["p_rain_yes"] = p_yes
    out["pred_label"] = ["YES" if v == 1 else "NO" for v in pred_yes]
    out["true_label"] = y_true.values

    print("\nSample predictions (first 15 rows):")
    print(out[["p_rain_yes", "pred_label", "true_label"]].head(15))

    # Accuracy
    correct = (out["pred_label"] == out["true_label"]).mean()
    print(f"\nAccuracy on {VAL_CSV}: {correct:.4f}")

    # Confusion counts
    tp = ((out["pred_label"] == "YES") & (out["true_label"] == "YES")).sum()
    tn = ((out["pred_label"] == "NO") & (out["true_label"] == "NO")).sum()
    fp = ((out["pred_label"] == "YES") & (out["true_label"] == "NO")).sum()
    fn = ((out["pred_label"] == "NO") & (out["true_label"] == "YES")).sum()
    print(f"Confusion (VAL): TP={tp} TN={tn} FP={fp} FN={fn}")

    # Save predictions for later inspection
    os.makedirs("reports", exist_ok=True)
    pred_path = os.path.join("reports", "val_predictions_latest.csv")
    out.to_csv(pred_path, index=False)
    print("Saved predictions to:", pred_path)



if __name__ == "__main__":
    main()
