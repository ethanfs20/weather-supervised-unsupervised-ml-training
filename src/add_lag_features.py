import pandas as pd
import numpy as np
import os

IN_TRAIN = "data/raw/real_rain_train.csv"
IN_VAL   = "data/raw/real_rain_val.csv"
OUT_DIR  = "data/processed"


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary: did it rain yesterday?
    df["rain_1d_ago"] = (df["precip_today_mm"].shift(1) >= 1.0).astype(int)

    # Rolling sums of rain
    df["rain_3d_sum"] = df["precip_today_mm"].rolling(3).sum().shift(1)
    df["rain_7d_sum"] = df["precip_today_mm"].rolling(7).sum().shift(1)

    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train = pd.read_csv(IN_TRAIN)
    val   = pd.read_csv(IN_VAL)

    train_lag = add_lags(train)
    val_lag   = add_lags(val)

    # Drop rows with incomplete lag history
    train_lag = train_lag.dropna().reset_index(drop=True)
    val_lag   = val_lag.dropna().reset_index(drop=True)

    train_path = f"{OUT_DIR}/real_rain_train_lag.csv"
    val_path   = f"{OUT_DIR}/real_rain_val_lag.csv"

    train_lag.to_csv(train_path, index=False)
    val_lag.to_csv(val_path, index=False)

    print("Saved:", train_path, "rows:", len(train_lag))
    print("Saved:", val_path, "rows:", len(val_lag))
    print("New columns:", [c for c in train_lag.columns if "rain_" in c])


if __name__ == "__main__":
    main()
