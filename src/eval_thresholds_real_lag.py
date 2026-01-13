import os, json, glob
import joblib
import pandas as pd

MODELS_DIR = "models"
VAL_CSV = "data/processed/real_rain_val_lag.csv"

def newest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime)

def main():
    model_path = newest_file(os.path.join(MODELS_DIR, "logreg_rain_*.joblib"))
    cols_path  = newest_file(os.path.join(MODELS_DIR, "feature_columns_*.json"))

    model = joblib.load(model_path)
    cols = json.load(open(cols_path, "r", encoding="utf-8"))

    df = pd.read_csv(VAL_CSV)
    X = df.drop(columns=["label"])[cols]
    y = df["label"].map({"NO": 0, "YES": 1}).astype(int).to_numpy()

    p = model.predict_proba(X)[:, 1]

    print("Model:", model_path)
    print("VAL:", VAL_CSV)
    print("YES rate:", (y == 1).mean().round(3))
    print("\nthreshold  accuracy  precision recall   TP   FP   FN   TN")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pred = (p >= t).astype(int)

        TP = int(((pred == 1) & (y == 1)).sum())
        TN = int(((pred == 0) & (y == 0)).sum())
        FP = int(((pred == 1) & (y == 0)).sum())
        FN = int(((pred == 0) & (y == 1)).sum())

        acc = (TP + TN) / len(y)
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0

        print(f"{t:0.2f}      {acc:0.4f}   {precision:0.3f}    {recall:0.3f}  {TP:4d} {FP:4d} {FN:4d} {TN:4d}")

if __name__ == "__main__":
    main()
