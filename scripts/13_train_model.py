"""
Phase 4 — Entraîne le modèle prédictif de probabilité d'incendie
(fire_detected, 1 jour x 1 wilaya) sur la zone forestière.

Choix méthodologiques :
- Ère VIIRS uniquement (2015+) : les labels MODIS (2001-2014, capteur
  ~5x moins sensible) ne sont pas comparables — les mélanger
  apprendrait au modèle un biais d'époque au lieu d'un signal météo.
- Split temporel : train 2015-2023, test 2024-2026. Un split aléatoire
  surestimerait la performance (forte autocorrélation jour à jour).
- Features limitées à ce que l'API de prévision Open-Meteo fournit
  (13 variables météo) + saisonnalité (mois encodé en sin/cos) +
  caractéristiques statiques de la wilaya. Aucune feature dérivée du
  label (pas de fréquence historique de feu) pour éviter les fuites.
- Modèle : HistGradientBoostingClassifier (gère les NaN nativement,
  rapide sur 150k lignes, probabilités raisonnablement calibrées sans
  rééquilibrage de classes).
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              roc_auc_score)

ML_TABLE = "../data/processed/ml_table_forest_zone_2000_2026.parquet"
OUT_MODEL = "../data/processed/model_fire_risk_v1.joblib"
OUT_META = "../data/processed/model_fire_risk_v1_meta.json"

WEATHER_FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "precipitation_sum", "rain_sum",
    "sunshine_duration", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "surface_pressure_mean",
]
STATIC_FEATURES = ["centroid_lat", "centroid_lon", "area_km2"]

TRAIN_END = "2023-12-31"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[WEATHER_FEATURES + STATIC_FEATURES].copy()
    month = df["date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["wilaya_id"] = df["wilaya_id"].astype("category")
    return X


def main():
    df = pd.read_parquet(ML_TABLE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2015-01-01") & (df["fire_data_coverage"])]
    print(f"Période VIIRS retenue : {df['date'].min().date()} -> {df['date'].max().date()} "
          f"({len(df)} lignes, {df['fire_detected'].mean()*100:.1f}% positives)")

    train = df[df["date"] <= TRAIN_END]
    test = df[df["date"] > TRAIN_END]
    X_train, y_train = build_features(train), train["fire_detected"]
    X_test, y_test = build_features(test), test["fire_detected"]
    print(f"Train : {len(train)} lignes (2015-2023) — Test : {len(test)} lignes (2024-2026)")

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=None, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0,
        categorical_features=["wilaya_id"],
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    proba_train = model.predict_proba(X_train)[:, 1]

    metrics = {
        "roc_auc_test": round(float(roc_auc_score(y_test, proba_test)), 4),
        "pr_auc_test": round(float(average_precision_score(y_test, proba_test)), 4),
        "brier_test": round(float(brier_score_loss(y_test, proba_test)), 4),
        "roc_auc_train": round(float(roc_auc_score(y_train, proba_train)), 4),
        "base_rate_test": round(float(y_test.mean()), 4),
        "n_train": len(train), "n_test": len(test),
    }
    print("\nMétriques (test 2024-2026, jamais vu à l'entraînement) :")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Calibration par décile de probabilité prédite
    calib = pd.DataFrame({"proba": proba_test, "obs": y_test.values})
    calib["decile"] = pd.qcut(calib["proba"], 10, labels=False, duplicates="drop")
    calib_table = calib.groupby("decile").agg(
        proba_moyenne=("proba", "mean"), taux_observe=("obs", "mean"), n=("obs", "size")
    ).round(4)
    print("\nCalibration (probabilité prédite vs taux de feu observé, par décile) :")
    print(calib_table.to_string())

    joblib.dump(model, OUT_MODEL)
    meta = {
        "version": "v1",
        "trained_on": "2015-01-01 -> 2023-12-31 (ère VIIRS, zone forestière, 36 wilayas)",
        "tested_on": "2024-01-01 -> 2026-07 (hors entraînement)",
        "features": list(X_train.columns),
        "metrics": metrics,
        "calibration": calib_table.reset_index().to_dict(orient="records"),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nÉcrit {OUT_MODEL} + {OUT_META}")


if __name__ == "__main__":
    main()
