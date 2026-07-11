"""
Mise à jour quotidienne incrémentale — conçue pour GitHub Actions.

Ne nécessite AUCUN fichier brut historique (data/raw/), uniquement les
artefacts versionnés :
  - wilayas.csv, wilayas_simplified.geojson
  - recurring_thermal_spots.csv        (liste des torchères, générée par 15)
  - ml_table_daily_wilaya_2000_2025.parquet  (historique figé)
  - fires_daily_wilaya_current_year.csv       (détections de l'année en cours)

Chaque exécution :
  1. Météo : re-télécharge toute la période courante (fin de l'historique
     figé -> hier) via l'API archive Open-Meteo, en requêtes groupées.
  2. Incendies : télécharge les 14 derniers jours FIRMS (VIIRS SP + NRT),
     jointure spatiale, filtre type + torchères, et fusionne dans le
     fichier de l'année en cours (les dates rechargées remplacent l'ancien).
  3. Reconstruit ml_table_current_year.parquet (météo + feux + wilayas).

Sortie committée par le workflow : fires_daily_wilaya_current_year.csv
+ ml_table_current_year.parquet (~1 Mo au total).
"""
import datetime
import io
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

PROCESSED = "../data/processed"
FROZEN_PARQUET = f"{PROCESSED}/ml_table_daily_wilaya_2000_2025.parquet"
FIRES_CURRENT = f"{PROCESSED}/fires_daily_wilaya_current_year.csv"
OUT_CURRENT = f"{PROCESSED}/ml_table_current_year.parquet"
WILAYAS_CSV = f"{PROCESSED}/wilayas.csv"
WILAYAS_GEOJSON = f"{PROCESSED}/wilayas_simplified.geojson"
RECURRING_SPOTS = f"{PROCESSED}/recurring_thermal_spots.csv"
FORECAST_LOG = f"{PROCESSED}/forecast_log.csv"
FORECAST_LOG_RETENTION_DAYS = 60

MAP_KEY = os.environ.get("FIRMS_MAP_KEY") or "3564558944d7ab736a51254db8be2620"
ALGERIA_BBOX = "-8.68,18.96,11.99,37.12"
FIRMS_SOURCES = ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_SP",
                 "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
FIRMS_LOOKBACK_DAYS = 14

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "precipitation_sum", "rain_sum",
    "sunshine_duration", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "surface_pressure_mean",
]


def http_get(url, timeout=60, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"    retry {attempt+1}/{retries} after {e!r} (wait {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Échec définitif : {url[:120]}")


def fetch_weather(wilayas, start, end):
    """Météo groupée pour toutes les wilayas (par lots de 29 pour rester
    sous les limites de taille de réponse de l'API)."""
    frames = []
    for i in range(0, len(wilayas), 29):
        batch = wilayas.iloc[i:i + 29]
        lats = ",".join(f"{v:.4f}" for v in batch["centroid_lat"])
        lons = ",".join(f"{v:.4f}" for v in batch["centroid_lon"])
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lats}&longitude={lons}"
            f"&start_date={start}&end_date={end}"
            f"&daily={','.join(DAILY_VARS)}&timezone=Africa%2FAlgiers"
        )
        data = json.loads(http_get(url))
        if not isinstance(data, list):
            data = [data]
        for w, loc in zip(batch.itertuples(), data):
            d = loc["daily"]
            df = pd.DataFrame({k: d[k] for k in ["time"] + DAILY_VARS})
            df["wilaya_id"] = w.wilaya_id
            df["wilaya_code"] = w.wilaya_code
            frames.append(df)
        time.sleep(2)
    out = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def fetch_firms_recent(today):
    """Détections FIRMS des 14 derniers jours, toutes sources VIIRS."""
    rows, header_cols = [], None
    dates = [today - datetime.timedelta(days=off) for off in (14, 9, 4)]
    for source in FIRMS_SOURCES:
        for d in dates:
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/"
                   f"{source}/{ALGERIA_BBOX}/5/{d.isoformat()}")
            try:
                text = http_get(url, timeout=20, retries=2)
            except RuntimeError:
                print(f"  (échec ignoré) {source} {d}")
                continue
            lines = text.strip().split("\n")
            if len(lines) < 2 or "," not in lines[0]:
                continue
            cols = lines[0].split(",")
            df = pd.read_csv(io.StringIO(text))
            if "type" not in df.columns:
                df["type"] = pd.NA
            rows.append(df)
            time.sleep(0.3)
    if not rows:
        return pd.DataFrame()
    fires = pd.concat(rows, ignore_index=True)
    fires["acq_date"] = pd.to_datetime(fires["acq_date"])
    fires = fires.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    return fires


def fetch_and_log_forecast(wilayas, today):
    """Archive la prévision à 7 jours émise aujourd'hui pour toutes les
    wilayas, afin de permettre un vrai backtesting prévision-vs-réel une
    fois que les jours cibles seront passés. Log glissant (fenêtre
    FORECAST_LOG_RETENTION_DAYS) pour ne pas grossir indéfiniment."""
    frames = []
    for i in range(0, len(wilayas), 29):
        batch = wilayas.iloc[i:i + 29]
        lats = ",".join(f"{v:.4f}" for v in batch["centroid_lat"])
        lons = ",".join(f"{v:.4f}" for v in batch["centroid_lon"])
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lats}&longitude={lons}"
            f"&daily={','.join(DAILY_VARS)}&forecast_days=7&timezone=Africa%2FAlgiers"
        )
        data = json.loads(http_get(url, timeout=30))
        if not isinstance(data, list):
            data = [data]
        for w, loc in zip(batch.itertuples(), data):
            d = loc["daily"]
            df = pd.DataFrame({k: d[k] for k in ["time"] + DAILY_VARS})
            df["wilaya_id"] = w.wilaya_id
            frames.append(df)
        time.sleep(1)
    new_log = pd.concat(frames, ignore_index=True).rename(columns={"time": "target_date"})
    new_log["issued_date"] = today.isoformat()

    if os.path.exists(FORECAST_LOG):
        existing = pd.read_csv(FORECAST_LOG)
        combined = pd.concat([existing, new_log], ignore_index=True)
        combined = combined.drop_duplicates(subset=["issued_date", "target_date", "wilaya_id"], keep="last")
    else:
        combined = new_log

    cutoff = (today - datetime.timedelta(days=FORECAST_LOG_RETENTION_DAYS)).isoformat()
    combined = combined[combined["issued_date"] >= cutoff]
    combined = combined.sort_values(["issued_date", "target_date", "wilaya_id"])
    combined.to_csv(FORECAST_LOG, index=False)
    print(f"  forecast_log.csv : {len(combined)} lignes ({combined['issued_date'].nunique()} jours d'émission)")


def main():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    wilayas = pd.read_csv(WILAYAS_CSV, dtype={"wilaya_code": str})
    frozen = pd.read_parquet(FROZEN_PARQUET)
    frozen["date"] = pd.to_datetime(frozen["date"])
    current_start = (frozen["date"].max() + pd.Timedelta(days=1)).date()
    print(f"Période courante : {current_start} -> {yesterday}")

    # --- 1. Météo ---
    print("Météo (archive Open-Meteo, requêtes groupées)...")
    weather = fetch_weather(wilayas, current_start.isoformat(), yesterday.isoformat())
    print(f"  {len(weather)} lignes ({weather['date'].nunique()} jours x {weather['wilaya_id'].nunique()} wilayas)")

    # --- 2. Incendies récents ---
    print("FIRMS (14 derniers jours)...")
    fires_new = fetch_firms_recent(today)
    print(f"  {len(fires_new)} détections brutes")

    fires_current = pd.read_csv(FIRES_CURRENT, parse_dates=["date"])
    if len(fires_new):
        geometry = [Point(xy) for xy in zip(fires_new["longitude"], fires_new["latitude"])]
        gdf = gpd.GeoDataFrame(fires_new.reset_index(drop=True), geometry=geometry, crs="EPSG:4326")
        wil_geo = gpd.read_file(WILAYAS_GEOJSON)
        joined = gpd.sjoin(gdf, wil_geo[["wilaya_id", "wilaya_code", "geometry"]],
                            how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        matched = joined[joined["wilaya_id"].notna()].copy()
        matched["wilaya_id"] = matched["wilaya_id"].astype(int)

        matched = matched[(matched["type"] == 0) | (matched["type"].isna())]
        spots = pd.read_csv(RECURRING_SPOTS)
        spot_index = set(zip(spots["lat_grid"], spots["lon_grid"]))
        keys = list(zip(matched["latitude"].round(2), matched["longitude"].round(2)))
        matched = matched[[k not in spot_index for k in keys]]
        print(f"  {len(matched)} détections après filtres (wilaya + type + torchères)")

        daily_new = matched.groupby(["wilaya_id", "wilaya_code", "acq_date"]).agg(
            nb_detections=("frp", "count"), frp_total=("frp", "sum"), frp_max=("frp", "max"),
        ).reset_index().rename(columns={"acq_date": "date"})
        daily_new["fire_detected"] = 1

        window_start = daily_new["date"].min()
        fires_current = pd.concat([
            fires_current[fires_current["date"] < window_start], daily_new,
        ], ignore_index=True).sort_values(["date", "wilaya_id"])
        fires_current = fires_current[fires_current["date"] >= pd.Timestamp(current_start)]
    fires_current.to_csv(FIRES_CURRENT, index=False)
    print(f"  fires_daily_wilaya_current_year.csv : {len(fires_current)} lignes")

    # --- 3. Table ML de l'année courante ---
    fires_small = fires_current[["wilaya_id", "date", "nb_detections", "frp_total", "frp_max", "fire_detected"]]
    merged = weather.merge(fires_small, on=["wilaya_id", "date"], how="left")
    merged["nb_detections"] = merged["nb_detections"].fillna(0).astype(int)
    merged["frp_total"] = merged["frp_total"].fillna(0.0).round(2)
    merged["frp_max"] = merged["frp_max"].fillna(0.0).round(2)
    merged["fire_detected"] = merged["fire_detected"].fillna(0).astype(int)
    merged = merged.merge(
        wilayas[["wilaya_id", "wilaya_name", "area_km2", "centroid_lat", "centroid_lon", "is_forest_zone"]],
        on="wilaya_id", how="left",
    )
    merged["fire_data_coverage"] = True
    merged = merged[list(frozen.columns)].sort_values(["wilaya_id", "date"]).reset_index(drop=True)
    merged.to_parquet(OUT_CURRENT, index=False)
    print(f"Écrit ml_table_current_year.parquet — {len(merged)} lignes "
          f"({merged['date'].min().date()} -> {merged['date'].max().date()}), "
          f"{int(merged['fire_detected'].sum())} jours-feu")

    # --- 4. Archivage de la prévision du jour (pour backtesting futur) ---
    print("Archivage de la prévision à 7 jours...")
    try:
        fetch_and_log_forecast(wilayas, today)
    except Exception as e:
        print(f"  (non bloquant) échec de l'archivage des prévisions : {e!r}")


if __name__ == "__main__":
    main()
