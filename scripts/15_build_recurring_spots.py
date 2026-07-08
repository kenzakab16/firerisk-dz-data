"""
Construit la liste des emplacements thermiques récurrents (torchères de
gaz/pétrole et autres sources fixes) à partir de l'historique brut
complet, et l'exporte en CSV versionné.

À lancer LOCALEMENT (nécessite data/raw/, ~800 Mo non versionnés).
Le job de mise à jour quotidienne (GitHub Actions) consomme ce CSV pour
filtrer les nouvelles détections sans avoir besoin de l'historique brut.
À relancer occasionnellement (ex. 1x/an) pour capter de nouvelles
installations industrielles.
"""
import glob

import pandas as pd

RAW_DIR = "../data/raw"
OUT = "../data/processed/recurring_thermal_spots.csv"
PERSISTENCE_THRESHOLD = 15

frames = []
for path in sorted(glob.glob(f"{RAW_DIR}/firms_*_algeria_*.csv")):
    df = pd.read_csv(path, usecols=lambda c: c in
                     {"latitude", "longitude", "acq_date", "type"})
    if "type" not in df.columns:
        df["type"] = pd.NA
    frames.append(df)
    print(f"  {path.split('/')[-1]}: {len(df)} lignes")

fires = pd.concat(frames, ignore_index=True)
fires = fires[(fires["type"] == 0) | (fires["type"].isna())]
fires["lat_grid"] = fires["latitude"].round(2)
fires["lon_grid"] = fires["longitude"].round(2)

persistence = fires.groupby(["lat_grid", "lon_grid"])["acq_date"].nunique()
recurring = persistence[persistence > PERSISTENCE_THRESHOLD].reset_index()
recurring.columns = ["lat_grid", "lon_grid", "n_days_detected"]
recurring.to_csv(OUT, index=False)
print(f"\nÉcrit {OUT} — {len(recurring)} emplacements récurrents "
      f"(> {PERSISTENCE_THRESHOLD} jours distincts sur 2000-2026)")
