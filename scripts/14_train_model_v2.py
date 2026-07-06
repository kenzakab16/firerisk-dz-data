"""
Phase 4 v2 — Ajoute les variables d'assèchement antérieur au modèle :
  - precip_sum_7d / precip_sum_30d : pluie cumulée sur les 7/30 jours
    précédents (strictement antérieurs : fenêtre décalée d'un jour)
  - temp_max_mean_7d : température max moyenne des 7 jours précédents
  - et0_sum_30d : évapotranspiration cumulée des 30 jours précédents
  - days_since_rain : jours écoulés depuis la dernière pluie >= 1 mm
    (plafonné à 90)

Toutes ces variables sont calculables au moment de la prédiction :
l'API de prévision Open-Meteo fournit les 31 jours passés + 7 jours
futurs en une seule requête (paramètre past_days).

Mêmes choix méthodologiques que v1 (ère VIIRS 2015+, split temporel
train 2015-2023 / test 2024-2026, aucune feature dérivée du label).
Les fenêtres glissantes sont calculées sur l'historique complet
(2000+), donc pleinement remplies dès le début du train.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              roc_auc_score)

ML_TABLE = "../data/processed/ml_table_forest_zone_2000_2026.parquet"
OUT_MODEL = "../data/processed/model_fire_risk_v2.joblib"
OUT_META = "../data/processed/model_fire_risk_v2_meta.json"

WEATHER_FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "precipitation_sum", "rain_sum",
    "sunshine_duration", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "surface_pressure_mean",
]
DRYNESS_FEATURES = ["precip_sum_7d", "precip_sum_30d", "temp_max_mean_7d",
                    "et0_sum_30d", "days_since_rain"]
STATIC_FEATURES = ["centroid_lat", "centroid_lon", "area_km2"]

TRAIN_END = "2023-12-31"
V1_METRICS = {"roc_auc_test": 0.7501, "pr_auc_test": 0.3324, "brier_test": 0.1051}


def add_dryness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fenêtres strictement antérieures (shift 1 avant le rolling) pour
    n'utiliser que l'information disponible la veille du jour prédit."""
    df = df.sort_values(["wilaya_id", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("wilaya_id", sort=False):
        g = g.copy()
        prev_precip = g["precipitation_sum"].shift(1)
        prev_temp = g["temperature_2m_max"].shift(1)
        prev_et0 = g["et0_fao_evapotranspiration"].shift(1)
        g["precip_sum_7d"] = prev_precip.rolling(7, min_periods=4).sum()
        g["precip_sum_30d"] = prev_precip.rolling(30, min_periods=15).sum()
        g["temp_max_mean_7d"] = prev_temp.rolling(7, min_periods=4).mean()
        g["et0_sum_30d"] = prev_et0.rolling(30, min_periods=15).sum()
        rained = (prev_precip >= 1.0).fillna(False)
        # cumcount depuis la dernière pluie (0 = il a plu hier)
        grp = rained.cumsum()
        g["days_since_rain"] = rained.groupby(grp).cumcount().clip(upper=90)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[WEATHER_FEATURES + DRYNESS_FEATURES + STATIC_FEATURES].copy()
    month = df["date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["wilaya_id"] = df["wilaya_id"].astype("category")
    return X


def main():
    df = pd.read_parquet(ML_TABLE)
    df["date"] = pd.to_datetime(df["date"])
    print("Calcul des variables d'assèchement (fenêtres sur l'historique complet 2000+)...")
    df = add_dryness_features(df)

    df = df[(df["date"] >= "2015-01-01") & (df["fire_data_coverage"])]
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
    metrics = {
        "roc_auc_test": round(float(roc_auc_score(y_test, proba_test)), 4),
        "pr_auc_test": round(float(average_precision_score(y_test, proba_test)), 4),
        "brier_test": round(float(brier_score_loss(y_test, proba_test)), 4),
        "roc_auc_train": round(float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])), 4),
        "base_rate_test": round(float(y_test.mean()), 4),
        "n_train": len(train), "n_test": len(test),
    }
    print("\nComparaison v1 -> v2 (test 2024-2026) :")
    for k in ["roc_auc_test", "pr_auc_test", "brier_test"]:
        print(f"  {k}: {V1_METRICS[k]} -> {metrics[k]}  (delta {metrics[k]-V1_METRICS[k]:+.4f})")

    print("\nImportance des variables d'assèchement (permutation, échantillon test) :")
    sample = X_test.sample(min(8000, len(X_test)), random_state=42)
    y_sample = y_test.loc[sample.index]
    imp = permutation_importance(model, sample, y_sample, n_repeats=3,
                                  random_state=42, scoring="roc_auc")
    imp_s = pd.Series(imp.importances_mean, index=sample.columns).sort_values(ascending=False)
    print(imp_s.head(12).round(4).to_string())

    calib = pd.DataFrame({"proba": proba_test, "obs": y_test.values})
    calib["decile"] = pd.qcut(calib["proba"], 10, labels=False, duplicates="drop")
    calib_table = calib.groupby("decile").agg(
        proba_moyenne=("proba", "mean"), taux_observe=("obs", "mean"), n=("obs", "size")
    ).round(4)

    joblib.dump(model, OUT_MODEL)
    meta = {
        "version": "v2",
        "changes_vs_v1": "ajout des variables d'assèchement antérieur (precip 7/30j, temp 7j, ET0 30j, jours sans pluie)",
        "trained_on": "2015-01-01 -> 2023-12-31 (ère VIIRS, zone forestière, 36 wilayas)",
        "tested_on": "2024-01-01 -> 2026-07 (hors entraînement)",
        "features": list(X_train.columns),
        "metrics": metrics,
        "v1_metrics": V1_METRICS,
        "calibration": calib_table.reset_index().to_dict(orient="records"),
        "top_permutation_importance": imp_s.head(12).round(4).to_dict(),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nÉcrit {OUT_MODEL} + {OUT_META}")


if __name__ == "__main__":
    main()
