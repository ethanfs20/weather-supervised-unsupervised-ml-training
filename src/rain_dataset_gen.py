#!/usr/bin/env python3
"""
rain_dataset_gen.py

Generates synthetic weather datasets for:
1) Supervised binary classification:
      "Is it going to rain tomorrow?" -> label in {"YES","NO"}
2) Unsupervised learning:
      Weather features only (clusterable "regimes"), with optional hidden regime id

Outputs (CSV):
  out/
    rain_supervised_train.csv
    rain_supervised_val.csv
    rain_unsupervised_train.csv
    rain_unsupervised_val.csv
    (optional) rain_unsup_hidden_regime_train.csv
    (optional) rain_unsup_hidden_regime_val.csv
    supervised_feature_scaler.csv
    unsupervised_feature_scaler.csv

Usage:
  python rain_dataset_gen.py --out_dir out --seed 42
  python rain_dataset_gen.py --sup_n 50000 --n_locations 20 --save_hidden_unsup_labels
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    out_dir: str = "out"
    seed: int = 42

    # Shared feature setup
    n_locations: int = 25
    years_span: int = 5  # sample day-of-year + year offset; not real dates but seasonal pattern
    include_location_latlon: bool = True

    # Supervised dataset
    sup_n: int = 50000
    sup_val_frac: float = 0.2
    label_noise_flip: float = 0.02  # flips YES<->NO fraction
    target_yes_rate: float = 0.35   # approximate rain frequency
    decision_noise: float = 0.35    # increases overlap between classes

    # Unsupervised dataset
    unsup_n: int = 50000
    unsup_val_frac: float = 0.2
    regimes: int = 5
    regime_spread: float = 1.0
    save_hidden_unsup_labels: bool = False


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def train_val_split(
    X: np.ndarray,
    y: np.ndarray | None,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr, X_val = X[tr_idx], X[val_idx]
    if y is None:
        return X_tr, X_val, None, None
    return X_tr, X_val, y[tr_idx], y[val_idx]


def standardize_fit_transform(X_tr: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_tr - mean) / std, (X_val - mean) / std, mean, std


def make_locations(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    # Roughly US-ish bounds for variety; just synthetic
    lats = rng.uniform(25.0, 49.0, size=cfg.n_locations)
    lons = rng.uniform(-124.0, -67.0, size=cfg.n_locations)

    # Each location has a baseline climate profile:
    # - coastal-ish vs inland-ish factor (arbitrary)
    coastal = rng.uniform(0.0, 1.0, size=cfg.n_locations)
    # - baseline humidity and rainfall tendency
    hum_base = 45 + 35 * coastal + rng.normal(0, 5, size=cfg.n_locations)     # %
    rain_base = 0.18 + 0.20 * coastal + rng.normal(0, 0.03, size=cfg.n_locations)  # probability-ish baseline

    df = pd.DataFrame({
        "location_id": np.arange(cfg.n_locations),
        "lat": lats,
        "lon": lons,
        "coastal": coastal,
        "hum_base": hum_base,
        "rain_base": rain_base
    })
    return df


def generate_weather_features(cfg: Config, rng: np.random.Generator, n: int, loc_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generates per-sample weather features:
      - day_of_year (1..365), sin/cos seasonality
      - temp_c
      - humidity_pct
      - pressure_hpa
      - wind_speed_ms
      - wind_gust_ms
      - cloud_cover_pct
      - precip_last_24h_mm
      - dewpoint_c (derived-ish)
      - location metadata (optional)
    Returns df_features, loc_ids, day_of_year
    """
    loc_ids = rng.integers(0, cfg.n_locations, size=n)
    day_of_year = rng.integers(1, 366, size=n)

    # Seasonal component (winter vs summer): sin peaks mid-year
    theta = 2 * np.pi * (day_of_year / 365.0)
    season_sin = np.sin(theta)
    season_cos = np.cos(theta)

    # Pull location baselines
    hum_base = loc_df["hum_base"].to_numpy()[loc_ids]
    rain_base = loc_df["rain_base"].to_numpy()[loc_ids]
    coastal = loc_df["coastal"].to_numpy()[loc_ids]

    # Temperature: seasonal + latitude effect
    lat = loc_df["lat"].to_numpy()[loc_ids]
    # Warmer at lower lat; colder at higher lat
    temp_c = (18 * season_sin) + (32 - 0.6 * (lat - 25)) + rng.normal(0, 3.0, size=n)

    # Humidity: higher for coastal + slightly higher in warm seasons, plus noise
    humidity = hum_base + 10 * np.clip(season_sin, -1, 1) + 8 * coastal + rng.normal(0, 8.0, size=n)
    humidity = np.clip(humidity, 5, 100)

    # Pressure: lower pressure tends to correlate with storminess
    pressure = 1016 - 6 * (rain_base * 10) - 5 * np.clip(humidity - 70, 0, 30) / 30 + rng.normal(0, 4.0, size=n)
    pressure = np.clip(pressure, 980, 1045)

    # Cloud cover: tied to humidity and pressure
    cloud = 20 + 0.8 * np.clip(humidity - 40, 0, 60) + 15 * np.clip(1012 - pressure, 0, 30) / 30 + rng.normal(0, 10.0, size=n)
    cloud = np.clip(cloud, 0, 100)

    # Wind + gusts: more wind when pressure lower
    wind = 2.0 + 0.25 * np.clip(1015 - pressure, 0, 35) + rng.normal(0, 1.2, size=n)
    wind = np.clip(wind, 0, None)

    gust = wind + rng.normal(2.0, 1.5, size=n)
    gust = np.clip(gust, wind, None)

    # Precip last 24h: sparse, but higher when conditions stormy
    # generate a base probability of having *some* precip
    p_precip = np.clip(rain_base + 0.004 * np.clip(humidity - 60, 0, 40) + 0.005 * np.clip(1012 - pressure, 0, 30), 0, 0.9)
    has_precip = rng.random(n) < p_precip
    precip_mm = np.where(has_precip, rng.gamma(shape=1.6, scale=4.0, size=n), 0.0)
    precip_mm = np.clip(precip_mm, 0, 80)

    # Dewpoint (rough-ish): temp - (100-humidity)/5
    dewpoint = temp_c - (100 - humidity) / 5.0 + rng.normal(0, 0.8, size=n)

    df = pd.DataFrame({
        "day_of_year": day_of_year,
        "season_sin": season_sin,
        "season_cos": season_cos,
        "temp_c": temp_c,
        "dewpoint_c": dewpoint,
        "humidity_pct": humidity,
        "pressure_hpa": pressure,
        "cloud_cover_pct": cloud,
        "wind_speed_ms": wind,
        "wind_gust_ms": gust,
        "precip_last_24h_mm": precip_mm,
        "location_id": loc_ids,
    })

    if cfg.include_location_latlon:
        df["lat"] = loc_df["lat"].to_numpy()[loc_ids]
        df["lon"] = loc_df["lon"].to_numpy()[loc_ids]
        df["coastal_idx"] = coastal

    return df, loc_ids, day_of_year


def supervised_label_rain_tomorrow(cfg: Config, rng: np.random.Generator, df: pd.DataFrame, loc_df: pd.DataFrame) -> np.ndarray:
    """
    Create P(rain tomorrow) using a logistic model with realistic influences:
      + high humidity, high cloud, low pressure, recent precip, higher dewpoint spread
      + seasonality and location baseline
    Then choose threshold to hit approximate target_yes_rate.
    """
    loc_ids = df["location_id"].to_numpy().astype(int)
    rain_base = loc_df["rain_base"].to_numpy()[loc_ids]

    humidity = df["humidity_pct"].to_numpy()
    cloud = df["cloud_cover_pct"].to_numpy()
    pressure = df["pressure_hpa"].to_numpy()
    wind = df["wind_speed_ms"].to_numpy()
    precip24 = df["precip_last_24h_mm"].to_numpy()
    dewpoint = df["dewpoint_c"].to_numpy()
    temp = df["temp_c"].to_numpy()
    season_sin = df["season_sin"].to_numpy()

    # Features that often correspond to rain
    # (Synthetic! but shaped so a model can learn)
    score = (
        -2.2
        + 0.030 * (humidity - 60)
        + 0.018 * (cloud - 50)
        + 0.060 * np.clip(1012 - pressure, 0, 40)
        + 0.045 * np.log1p(precip24)
        + 0.020 * (dewpoint - (temp - 10))   # higher dewpoint relative to temp => more moisture
        + 0.010 * wind
        + 0.55 * rain_base
        + 0.25 * np.clip(season_sin, -0.2, 1.0)  # slight season effect
    )

    # Add noise to make it non-trivial
    score = score + rng.normal(0, cfg.decision_noise, size=score.shape[0])

    prob = sigmoid(score)

    # Choose threshold so YES rate ~ target_yes_rate
    thresh = np.quantile(prob, 1.0 - cfg.target_yes_rate)
    y = (prob >= thresh).astype(np.int32)  # 1 YES, 0 NO

    # Optional label flips for realism
    if cfg.label_noise_flip > 0:
        flips = rng.random(y.shape[0]) < cfg.label_noise_flip
        y[flips] = 1 - y[flips]

    return y


def df_to_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=np.float32)


def save_scaler(out_dir: str, filename: str, feature_cols: list[str], mean: np.ndarray, std: np.ndarray) -> None:
    scaler_df = pd.DataFrame({"feature": feature_cols, "mean": mean, "std": std})
    scaler_df.to_csv(os.path.join(out_dir, filename), index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate YES/NO 'rain tomorrow' supervised + unsupervised datasets.")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_locations", type=int, default=25)
    p.add_argument("--years_span", type=int, default=5)
    p.add_argument("--include_location_latlon", action="store_true")

    p.add_argument("--sup_n", type=int, default=50000)
    p.add_argument("--sup_val_frac", type=float, default=0.2)
    p.add_argument("--label_noise_flip", type=float, default=0.02)
    p.add_argument("--target_yes_rate", type=float, default=0.35)
    p.add_argument("--decision_noise", type=float, default=0.35)

    p.add_argument("--unsup_n", type=int, default=50000)
    p.add_argument("--unsup_val_frac", type=float, default=0.2)
    p.add_argument("--regimes", type=int, default=5)
    p.add_argument("--regime_spread", type=float, default=1.0)
    p.add_argument("--save_hidden_unsup_labels", action="store_true")

    args = p.parse_args()

    cfg = Config(
        out_dir=args.out_dir,
        seed=args.seed,
        n_locations=args.n_locations,
        years_span=args.years_span,
        include_location_latlon=args.include_location_latlon,
        sup_n=args.sup_n,
        sup_val_frac=args.sup_val_frac,
        label_noise_flip=args.label_noise_flip,
        target_yes_rate=args.target_yes_rate,
        decision_noise=args.decision_noise,
        unsup_n=args.unsup_n,
        unsup_val_frac=args.unsup_val_frac,
        regimes=args.regimes,
        regime_spread=args.regime_spread,
        save_hidden_unsup_labels=args.save_hidden_unsup_labels,
    )

    ensure_out_dir(cfg.out_dir)
    rng = np.random.default_rng(cfg.seed)

    loc_df = make_locations(cfg, rng)

    # -------------------------
    # SUPERVISED
    # -------------------------
    sup_df, _, _ = generate_weather_features(cfg, rng, cfg.sup_n, loc_df)
    y = supervised_label_rain_tomorrow(cfg, rng, sup_df, loc_df)

    # Feature set used for modeling
    feature_cols = [c for c in sup_df.columns if c not in []]  # all columns are features here

    X = df_to_matrix(sup_df, feature_cols)

    X_tr, X_val, y_tr, y_val = train_val_split(X, y, cfg.sup_val_frac, rng)
    X_tr_z, X_val_z, mean, std = standardize_fit_transform(X_tr, X_val)
    save_scaler(cfg.out_dir, "supervised_feature_scaler.csv", feature_cols, mean, std)

    sup_tr = pd.DataFrame(X_tr_z, columns=feature_cols)
    sup_val = pd.DataFrame(X_val_z, columns=feature_cols)
    sup_tr["label"] = np.where(y_tr == 1, "YES", "NO")
    sup_val["label"] = np.where(y_val == 1, "YES", "NO")

    sup_tr.to_csv(os.path.join(cfg.out_dir, "rain_supervised_train.csv"), index=False)
    sup_val.to_csv(os.path.join(cfg.out_dir, "rain_supervised_val.csv"), index=False)

    # -------------------------
    # UNSUPERVISED (regimes)
    # -------------------------
    # Create "weather regimes" by sampling a regime id and nudging certain features
    unsup_df, _, _ = generate_weather_features(cfg, rng, cfg.unsup_n, loc_df)

    regimes = cfg.regimes
    regime_id = rng.integers(0, regimes, size=cfg.unsup_n)

    # Regime effects (synthetic): e.g. stormy, dry-high-pressure, humid-coastal, windy-front, cold-snap
    # Apply shifts to make it clusterable.
    def apply_shift(col: str, shift: np.ndarray) -> None:
        unsup_df[col] = unsup_df[col].to_numpy() + shift

    r = regime_id.astype(np.int32)
    # Create per-sample shifts
    humidity_shift = np.select(
        [r == 0, r == 1, r == 2, r == 3, r == 4],
        [15, -10, 8, 5, -5],
        default=0
    )
    pressure_shift = np.select(
        [r == 0, r == 1, r == 2, r == 3, r == 4],
        [-10, 12, -4, -6, 8],
        default=0
    )
    cloud_shift = np.select(
        [r == 0, r == 1, r == 2, r == 3, r == 4],
        [20, -15, 10, 5, -10],
        default=0
    )
    wind_shift = np.select(
        [r == 0, r == 1, r == 2, r == 3, r == 4],
        [2.5, -0.5, 1.0, 3.5, -1.0],
        default=0
    )
    temp_shift = np.select(
        [r == 0, r == 1, r == 2, r == 3, r == 4],
        [-2, 3, 1, -5, -8],
        default=0
    )

    # Add noise scaled by regime_spread (bigger spread => more overlap)
    s = cfg.regime_spread
    apply_shift("humidity_pct", humidity_shift + rng.normal(0, 4.0 * s, size=cfg.unsup_n))
    apply_shift("pressure_hpa", pressure_shift + rng.normal(0, 2.5 * s, size=cfg.unsup_n))
    apply_shift("cloud_cover_pct", cloud_shift + rng.normal(0, 6.0 * s, size=cfg.unsup_n))
    apply_shift("wind_speed_ms", wind_shift + rng.normal(0, 0.8 * s, size=cfg.unsup_n))
    apply_shift("temp_c", temp_shift + rng.normal(0, 2.0 * s, size=cfg.unsup_n))
    # Recompute dewpoint loosely after temp/humidity shift
    unsup_df["dewpoint_c"] = unsup_df["temp_c"] - (100 - unsup_df["humidity_pct"]) / 5.0 + rng.normal(0, 0.6, size=cfg.unsup_n)

    # Clip ranges
    unsup_df["humidity_pct"] = np.clip(unsup_df["humidity_pct"], 5, 100)
    unsup_df["pressure_hpa"] = np.clip(unsup_df["pressure_hpa"], 980, 1045)
    unsup_df["cloud_cover_pct"] = np.clip(unsup_df["cloud_cover_pct"], 0, 100)
    unsup_df["wind_speed_ms"] = np.clip(unsup_df["wind_speed_ms"], 0, None)

    unsup_feature_cols = list(unsup_df.columns)
    X_u = df_to_matrix(unsup_df, unsup_feature_cols)

    X_u_tr, X_u_val, y_u_tr, y_u_val = train_val_split(X_u, regime_id, cfg.unsup_val_frac, rng)
    X_u_tr_z, X_u_val_z, mean_u, std_u = standardize_fit_transform(X_u_tr, X_u_val)
    save_scaler(cfg.out_dir, "unsupervised_feature_scaler.csv", unsup_feature_cols, mean_u, std_u)

    unsup_tr = pd.DataFrame(X_u_tr_z, columns=unsup_feature_cols)
    unsup_val = pd.DataFrame(X_u_val_z, columns=unsup_feature_cols)

    unsup_tr.to_csv(os.path.join(cfg.out_dir, "rain_unsupervised_train.csv"), index=False)
    unsup_val.to_csv(os.path.join(cfg.out_dir, "rain_unsupervised_val.csv"), index=False)

    if cfg.save_hidden_unsup_labels:
        pd.Series(y_u_tr, name="regime_id").to_csv(os.path.join(cfg.out_dir, "rain_unsup_hidden_regime_train.csv"), index=False)
        pd.Series(y_u_val, name="regime_id").to_csv(os.path.join(cfg.out_dir, "rain_unsup_hidden_regime_val.csv"), index=False)

    # -------------------------
    # Quick summary
    # -------------------------
    yes_rate_train = (sup_tr["label"] == "YES").mean()
    print("Wrote files to:", os.path.abspath(cfg.out_dir))
    print(f"Supervised train: {sup_tr.shape} | YES rate: {yes_rate_train:.3f}")
    print(f"Supervised val  : {sup_val.shape}")
    print(f"Unsupervised train: {unsup_tr.shape}")
    print(f"Unsupervised val  : {unsup_val.shape}")
    if cfg.save_hidden_unsup_labels:
        print("Saved hidden unsupervised regime labels (for evaluation).")


if __name__ == "__main__":
    main()
