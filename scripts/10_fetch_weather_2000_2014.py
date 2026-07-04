"""
Étend la météo en arrière : récupère 2000-01-01 -> 2014-12-31 pour les
58 wilayas (Open-Meteo Archive / ERA5, données réelles disponibles
depuis 1940 -- aucune donnée simulée).
"""
import os
import time
import urllib.request
import urllib.error
import json
import pandas as pd

WILAYAS_CSV = "../data/processed/wilayas.csv"
CHECKPOINT_DIR = "../data/raw/weather_by_wilaya_2000_2014"
OUT_CSV = "../data/processed/weather_2000_2014.csv"
START_DATE = "2000-01-01"
END_DATE = "2014-12-31"

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "precipitation_sum", "rain_sum",
    "sunshine_duration", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "surface_pressure_mean",
]

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_wilaya(lat, lon, retries=8):
    params = (
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily={','.join(DAILY_VARS)}"
        f"&timezone=Africa%2FAlgiers"
    )
    url = f"{BASE_URL}?{params}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            wait = 20 * (attempt + 1) if e.code == 429 else 5 * (attempt + 1)
            print(f"    retry {attempt+1}/{retries} after HTTP {e.code} (wait {wait}s)")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt+1}/{retries} after {e!r} (wait {wait}s)")
            time.sleep(wait)
    return None


def main():
    print(f"Période : {START_DATE} -> {END_DATE} (données réelles ERA5, aucune simulation)")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    wilayas = pd.read_csv(WILAYAS_CSV)

    for i, row in wilayas.iterrows():
        wid = row["wilaya_id"]
        ckpt_path = f"{CHECKPOINT_DIR}/wilaya_{wid:02d}.csv"
        if os.path.exists(ckpt_path):
            print(f"[{i+1}/{len(wilayas)}] {row['wilaya_name']} — déjà en cache, skip")
            continue
        print(f"[{i+1}/{len(wilayas)}] {row['wilaya_name']} ({row['centroid_lat']:.2f}, {row['centroid_lon']:.2f})")
        data = fetch_wilaya(row["centroid_lat"], row["centroid_lon"])
        if data is None:
            print(f"  ECHEC définitif pour wilaya {wid} — on continue")
            time.sleep(10)
            continue
        daily = data["daily"]
        df = pd.DataFrame({k: daily[k] for k in ["time"] + DAILY_VARS})
        df["wilaya_id"] = wid
        df["wilaya_code"] = row["wilaya_code"]
        df.to_csv(ckpt_path, index=False)
        time.sleep(6)

    frames = []
    for i, row in wilayas.iterrows():
        wid = row["wilaya_id"]
        ckpt_path = f"{CHECKPOINT_DIR}/wilaya_{wid:02d}.csv"
        if os.path.exists(ckpt_path):
            frames.append(pd.read_csv(ckpt_path))
    if not frames:
        print("Aucune donnée récupérée.")
        return
    combined = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    cols = ["wilaya_id", "wilaya_code", "date"] + DAILY_VARS
    combined = combined[cols].sort_values(["wilaya_id", "date"])
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nÉcrit {OUT_CSV} — {len(combined)} lignes "
          f"({combined['wilaya_id'].nunique()} wilayas, {combined['date'].min()} -> {combined['date'].max()})")


if __name__ == "__main__":
    main()
