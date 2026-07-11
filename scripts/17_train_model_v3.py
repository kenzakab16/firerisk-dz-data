"""
Phase 4 v3 — Deux leviers testés séparément, sur le harnais d'évaluation
de la v1 (ère VIIRS 2015+, split temporel train 2015-2023 / test 2024-2026) :

  1. Proxys d'activité humaine dérivables des données existantes :
     jour de la semaine (sin/cos) + indicateur week-end. 90-95% des feux
     en Algérie sont d'origine humaine (brûlage agricole, mégots,
     décharges) — l'activité humaine varie selon le jour.
  2. Pondération de classe (class_weight="balanced") pour le déséquilibre
     (~14% de positifs). NB : la pondération modifie surtout la
     calibration des probabilités ; l'effet sur le classement (PR-AUC)
     doit être mesuré, pas supposé.

En plus des métriques de classement, on mesure une métrique
OPÉRATIONNELLE : avec un budget de surveillance de k wilayas par jour
(celles au score le plus élevé), quelle part des feux du jour est
couverte (rappel@k) et quelle part des alertes est justifiée
(précision@k) ? C'est le vrai cas d'usage Protection Civile.

Variables reportées à une phase de collecte dédiée (pas de données
inventées) : NDVI/état de la végétation, occupation du sol, relief,
densité de population, distance aux routes/décharges.
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
OUT_MODEL = "../data/processed/model_fire_risk_v3.joblib"
OUT_META = "../data/processed/model_fire_risk_v3_meta.json"

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
ALERT_BUDGET = 8  # wilayas surveillées par jour


def build_features(df, with_dow=False):
    X = df[WEATHER_FEATURES + STATIC_FEATURES].copy()
    month = df["date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    if with_dow:
        dow = df["date"].dt.dayofweek
        X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        # Week-end algérien : vendredi (4) - samedi (5)
        X["is_weekend"] = dow.isin([4, 5]).astype(int)
    X["wilaya_id"] = df["wilaya_id"].astype("category")
    return X


def alert_budget_metrics(test_df, proba, k=ALERT_BUDGET):
    """Rappel/précision moyens par jour avec un budget de k wilayas alertées."""
    d = test_df[["date", "fire_detected"]].copy()
    d["proba"] = proba
    recalls, precisions = [], []
    for _, g in d.groupby("date"):
        top = g.nlargest(k, "proba")
        fires = g["fire_detected"].sum()
        if fires > 0:
            recalls.append(top["fire_detected"].sum() / fires)
        precisions.append(top["fire_detected"].mean())
    return float(np.mean(recalls)), float(np.mean(precisions))


def fit_eval(name, train, test, with_dow, class_weight):
    X_tr = build_features(train, with_dow)
    X_te = build_features(test, with_dow)
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0,
        categorical_features=["wilaya_id"], class_weight=class_weight,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    model.fit(X_tr, train["fire_detected"])
    proba = model.predict_proba(X_te)[:, 1]
    rec_k, prec_k = alert_budget_metrics(test, proba)
    metrics = {
        "roc_auc": round(float(roc_auc_score(test["fire_detected"], proba)), 4),
        "pr_auc": round(float(average_precision_score(test["fire_detected"], proba)), 4),
        "brier": round(float(brier_score_loss(test["fire_detected"], proba)), 4),
        f"recall@{ALERT_BUDGET}": round(rec_k, 4),
        f"precision@{ALERT_BUDGET}": round(prec_k, 4),
    }
    print(f"  {name:38s} " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return model, metrics, X_te


def main():
    df = pd.read_parquet(ML_TABLE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2015-01-01") & (df["fire_data_coverage"])]
    train = df[df["date"] <= TRAIN_END]
    test = df[df["date"] > TRAIN_END]
    print(f"Train {len(train)} (2015-2023) / Test {len(test)} (2024-2026), "
          f"base rate test {test['fire_detected'].mean()*100:.1f}%\n")

    results = {}
    print("Variantes (mêmes hyperparamètres, un levier à la fois) :")
    mA, resA, XA = fit_eval("A  v1 (référence météo)", train, test, False, None)
    mB, resB, XB = fit_eval("B  A + jour de semaine/week-end", train, test, True, None)
    mC, resC, XC = fit_eval("C  A + class_weight=balanced", train, test, False, "balanced")
    mD, resD, XD = fit_eval("D  B + class_weight=balanced", train, test, True, "balanced")
    results = {"A_v1_ref": resA, "B_dow": resB, "C_balanced": resC, "D_dow_balanced": resD}

    # Choix : meilleur PR-AUC ; en cas de quasi-égalité (<0,005), le plus simple
    ranked = sorted(results.items(), key=lambda kv: kv[1]["pr_auc"], reverse=True)
    best_name = ranked[0][0]
    print(f"\nMeilleure variante par PR-AUC : {best_name}")

    variants = {"A_v1_ref": (mA, False, XA), "B_dow": (mB, True, XB),
                "C_balanced": (mC, False, XC), "D_dow_balanced": (mD, True, XD)}
    best_model, best_dow, best_Xte = variants[best_name]

    # Importance par permutation du modèle retenu (pour l'explicabilité dashboard)
    print("\nImportance par permutation (échantillon test, ROC-AUC) :")
    sample_idx = best_Xte.sample(min(8000, len(best_Xte)), random_state=42).index
    imp = permutation_importance(best_model, best_Xte.loc[sample_idx],
                                  test["fire_detected"].loc[sample_idx],
                                  n_repeats=5, random_state=42, scoring="roc_auc")
    imp_s = pd.Series(imp.importances_mean, index=best_Xte.columns).sort_values(ascending=False)
    print(imp_s.round(4).to_string())

    joblib.dump(best_model, OUT_MODEL)
    meta = {
        "version": "v3",
        "variant": best_name,
        "with_dow_features": best_dow,
        "trained_on": "2015-01-01 -> 2023-12-31 (ère VIIRS, zone forestière)",
        "tested_on": "2024-01-01 -> 2026-07 (hors entraînement)",
        "features": list(best_Xte.columns),
        "metrics": {**results[best_name],
                     "roc_auc_test": results[best_name]["roc_auc"],
                     "pr_auc_test": results[best_name]["pr_auc"],
                     "base_rate_test": round(float(test["fire_detected"].mean()), 4)},
        "all_variants": results,
        "permutation_importance": imp_s.round(5).to_dict(),
        "alert_budget": ALERT_BUDGET,
        "deferred_features": "NDVI/végétation, occupation du sol, relief, densité de population, "
                             "distance routes/décharges — nécessitent une phase de collecte dédiée",
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nÉcrit {OUT_MODEL} + {OUT_META} (variante {best_name})")


if __name__ == "__main__":
    main()
