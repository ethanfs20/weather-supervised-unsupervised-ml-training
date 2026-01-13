from __future__ import annotations

from datetime import date
import os

import numpy as np
import pandas as pd
import meteostat as ms


# -----------------------
# CONFIG
# -----------------------
OUT_DIR = "data/raw"

# Use a station directly (more reliable than interpolation for some parameters)
# Chicago O'Hare Airport station id on Meteostat: 72530  :contentReference[oaicite:1]{index=1}
STATION_ID = "72530"

START = date(2018, 1, 1)
END   = date(2024, 12, 31)

# Label: rain tomorrow if tomorrow precip >= this threshold (mm)
RAIN_MM_THRESHOLD = 1.0

VAL_FRAC = 0.2
SEED = 42


def train_val_split(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(round(len(df) * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Fetch daily data for the station
    ts = ms.daily(ms.Station(id=STATION_ID), START, END)
    df = ts.fetch().reset_index().rename(columns={"time": "date"})

    # Expected Meteostat daily columns often include: tavg, tmin, tmax, prcp, wspd, pres
    for c in ["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df.sort_values("date").reset_index(drop=True)

    # Build label from tomorrow's precipitation
    df["prcp_tomorrow"] = df["prcp"].shift(-1)
    df = df.dropna(subset=["prcp_tomorrow"]).copy()
    df["label"] = np.where(df["prcp_tomorrow"] >= RAIN_MM_THRESHOLD, "YES", "NO")

    # Seasonality
    df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
    theta = 2 * np.pi * (df["day_of_year"] / 365.0)
    df["season_sin"] = np.sin(theta)
    df["season_cos"] = np.cos(theta)

    # Build feature table
    out = pd.DataFrame({
        "day_of_year": df["day_of_year"].astype(int),
        "season_sin": df["season_sin"].astype(float),
        "season_cos": df["season_cos"].astype(float),

        # temps
        "temp_c": df["tavg"].astype(float),
        "temp_min_c": df["tmin"].astype(float),
        "temp_max_c": df["tmax"].astype(float),

        # other weather
        "pressure_hpa": df["pres"].astype(float),
        "wind_speed_kmh": df["wspd"].astype(float),
        "precip_today_mm": df["prcp"].astype(float),

        "label": df["label"].astype(str),
    })

    # If temp_c (tavg) is missing, derive it from min/max
    derived_temp = (out["temp_min_c"] + out["temp_max_c"]) / 2.0
    out["temp_c"] = out["temp_c"].fillna(derived_temp)

    # Print missingness so you can see what's real vs sparse
    print("\nStation:", STATION_ID)
    print("Missingness by column (fraction NaN):")
    print(out.isna().mean().sort_values(ascending=False))
    print("\nRows before cleaning:", len(out))

    # Minimal required fields to learn something
    out = out.dropna(subset=["temp_c", "precip_today_mm", "label"]).copy()
    print("Rows after minimal dropna:", len(out))

    # Optional: impute sparse columns so you don't lose rows
    for c in ["pressure_hpa", "wind_speed_kmh", "temp_min_c", "temp_max_c"]:
        if c in out.columns:
            med = out[c].median()
            if pd.isna(med):
                med = 0.0
            out[c] = out[c].fillna(med)

    out["precip_today_mm"] = out["precip_today_mm"].clip(lower=0)

    # Split and save
    train, val = train_val_split(out, VAL_FRAC, SEED)

    train_path = os.path.join(OUT_DIR, "real_rain_train.csv")
    val_path = os.path.join(OUT_DIR, "real_rain_val.csv")
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    print("\nSaved:", train_path, "rows:", len(train))
    print("Saved:", val_path, "rows:", len(val))
    if len(train) > 0:
        print("YES rate (train):", round((train["label"] == "YES").mean(), 3))
    print("Columns:", list(train.columns))


if __name__ == "__main__":
    main()
