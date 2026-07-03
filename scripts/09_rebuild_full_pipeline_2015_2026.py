"""
Reconstruit la chaîne complète sur 2015-2026 :
  1. Fusionne toutes les sources FIRMS (SP 2015-2023, SP+NRT 2024-2026)
  2. Jointure spatiale (point-in-polygon) vers les 58 wilayas
  3. Filtre type=0 (feu de végétation) OU type manquant (NRT, pas de champ type)
  4. Filtre de persistance spatiale (torchères), recalculé sur 2015-2026 entier
  5. Fusion avec weather_2015_2026.csv -> table ML finale
  6. Filtre zone forestière (36 wilayas)

Les flux NRT (2026-04-28 -> aujourd'hui) n'ont pas de colonne `type`.
On les garde (type=NaN traité comme "à vérifier") et on compte sur le
filtre de persistance -- recalculé sur l'historique complet -- pour
exclure les torchères déjà identifiées comme sources récurrentes.
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

RAW_DIR = "../data/raw"
WILAYAS_GEOJSON = "../data/processed/wilayas_simplified.geojson"
WILAYAS_CSV = "../data/processed/wilayas.csv"
WEATHER_CSV = "../data/processed/weather_2015_2026.csv"

OUT_FIRES_RAW = "../data/processed/fires_raw_with_wilaya_2015_2026.csv"
OUT_FIRES_DAILY = "../data/processed/fires_daily_wilaya_2015_2026.csv"
OUT_ML = "../data/processed/ml_table_daily_wilaya_2015_2026.csv"
OUT_ML_PARQUET = "../data/processed/ml_table_daily_wilaya_2015_2026.parquet"
OUT_ML_FOREST = "../data/processed/ml_table_forest_zone_2015_2026.csv"
OUT_ML_FOREST_PARQUET = "../data/processed/ml_table_forest_zone_2015_2026.parquet"

SOURCES = [
    ("firms_viirs_snpp_sp_algeria_2015_2023.csv", "SNPP", "SP"),
    ("firms_viirs_noaa20_sp_algeria_2015_2023.csv", "NOAA20", "SP"),
    ("firms_viirs_snpp_sp_algeria_2024_2026.csv", "SNPP", "SP"),
    ("firms_viirs_snpp_nrt_algeria_2024_2026.csv", "SNPP", "NRT"),
    ("firms_viirs_noaa20_sp_algeria_2024_2026.csv", "NOAA20", "SP"),
    ("firms_viirs_noaa20_nrt_algeria_2024_2026.csv", "NOAA20", "NRT"),
    ("firms_viirs_noaa21_nrt_algeria_2024_2026.csv", "NOAA21", "NRT"),
]

PERSISTENCE_THRESHOLD = 15

NON_FOREST_WILAYAS = {
    "Adrar", "Béchar", "Tamanrasset", "Ouargla", "Illizi", "Tindouf",
    "El Oued", "Ghardaia", "El Bayadh", "Béni Abbès", "In Salah",
    "In Guezzam", "Timimoune", "Bordj Badji Mokhtar", "Touggourt",
    "Djanet", "El M'Ghair", "El Menia", "Ouled Djellal", "Naâma",
    "Laghouat", "Biskra",
}


def main():
    print("=== 1. Fusion des sources FIRMS ===")
    frames = []
    for fname, sat_label, quality in SOURCES:
        path = f"{RAW_DIR}/{fname}"
        try:
            df = pd.read_csv(path, parse_dates=["acq_date"])
        except FileNotFoundError:
            print(f"  (absent, ignoré) {fname}")
            continue
        df["source_satellite"] = sat_label
        df["data_quality"] = quality
        if "type" not in df.columns:
            df["type"] = pd.NA
        frames.append(df)
        print(f"  {fname}: {len(df)} lignes")

    fires = pd.concat(frames, ignore_index=True)
    fires = fires.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    print(f"Total après dédoublonnage : {len(fires)}")

    print("\n=== 2. Jointure spatiale ===")
    wilayas = gpd.read_file(WILAYAS_GEOJSON)
    fires = fires.reset_index(drop=True)
    geometry = [Point(xy) for xy in zip(fires["longitude"], fires["latitude"])]
    fires_gdf = gpd.GeoDataFrame(fires, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(fires_gdf, wilayas[["wilaya_id", "wilaya_code", "wilaya_name", "geometry"]],
                        how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    n_matched = joined["wilaya_id"].notna().sum()
    print(f"  {n_matched}/{len(joined)} détections rattachées à une wilaya")

    out = joined.drop(columns=["geometry", "index_right"])
    out.to_csv(OUT_FIRES_RAW, index=False)
    print(f"Écrit {OUT_FIRES_RAW}")

    print("\n=== 3. Filtre type (végétation ou NRT sans info type) ===")
    matched = out[out["wilaya_id"].notna()].copy()
    matched["wilaya_id"] = matched["wilaya_id"].astype(int)
    is_vegetation_or_unknown = (matched["type"] == 0) | (matched["type"].isna())
    matched = matched[is_vegetation_or_unknown].copy()
    print(f"  {len(matched)} détections conservées (type=0 ou NRT sans champ type)")

    print("\n=== 4. Filtre de persistance spatiale (torchères) ===")
    matched["lat_grid"] = matched["latitude"].round(2)
    matched["lon_grid"] = matched["longitude"].round(2)
    persistence = matched.groupby(["lat_grid", "lon_grid"])["acq_date"].nunique()
    recurring = persistence[persistence > PERSISTENCE_THRESHOLD].index
    n_before = len(matched)
    matched = matched.set_index(["lat_grid", "lon_grid"])
    matched = matched[~matched.index.isin(recurring)].reset_index()
    print(f"  {n_before - len(matched)} détections exclues ({len(recurring)} emplacements récurrents > {PERSISTENCE_THRESHOLD}j)")

    daily = matched.groupby(["wilaya_id", "wilaya_code", "acq_date"]).agg(
        nb_detections=("frp", "count"),
        frp_total=("frp", "sum"),
        frp_max=("frp", "max"),
    ).reset_index().rename(columns={"acq_date": "date"})
    daily["fire_detected"] = 1
    daily.to_csv(OUT_FIRES_DAILY, index=False)
    print(f"Écrit {OUT_FIRES_DAILY} — {len(daily)} lignes")

    print("\n=== 5. Fusion météo + incendies ===")
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    fires_small = daily[["wilaya_id", "date", "nb_detections", "frp_total", "frp_max", "fire_detected"]]
    fires_small["date"] = pd.to_datetime(fires_small["date"])

    wilayas_df = pd.read_csv(WILAYAS_CSV)
    wilayas_df["is_forest_zone"] = ~wilayas_df["wilaya_name"].isin(NON_FOREST_WILAYAS)
    wilayas_df.to_csv(WILAYAS_CSV, index=False)

    merged = weather.merge(fires_small, on=["wilaya_id", "date"], how="left")
    merged["nb_detections"] = merged["nb_detections"].fillna(0).astype(int)
    merged["frp_total"] = merged["frp_total"].fillna(0.0).round(2)
    merged["frp_max"] = merged["frp_max"].fillna(0.0).round(2)
    merged["fire_detected"] = merged["fire_detected"].fillna(0).astype(int)
    merged = merged.merge(
        wilayas_df[["wilaya_id", "wilaya_name", "area_km2", "centroid_lat", "centroid_lon", "is_forest_zone"]],
        on="wilaya_id", how="left",
    )

    cols_order = [
        "date", "wilaya_id", "wilaya_code", "wilaya_name",
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
        "precipitation_sum", "rain_sum",
        "sunshine_duration", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "surface_pressure_mean",
        "area_km2", "centroid_lat", "centroid_lon", "is_forest_zone",
        "nb_detections", "frp_total", "frp_max", "fire_detected",
    ]
    merged = merged[cols_order].sort_values(["wilaya_id", "date"]).reset_index(drop=True)
    merged.to_csv(OUT_ML, index=False)
    merged.to_parquet(OUT_ML_PARQUET, index=False)
    print(f"Écrit {OUT_ML} — {len(merged)} lignes "
          f"({merged['wilaya_id'].nunique()} wilayas x {merged['date'].nunique()} jours, "
          f"{merged['date'].min().date()} -> {merged['date'].max().date()})")

    print("\n=== 6. Filtre zone forestière ===")
    forest = merged[merged["is_forest_zone"]].copy()
    forest.to_csv(OUT_ML_FOREST, index=False)
    forest.to_parquet(OUT_ML_FOREST_PARQUET, index=False)
    print(f"Écrit {OUT_ML_FOREST} — {len(forest)} lignes "
          f"({forest['wilaya_id'].nunique()} wilayas)")
    print(f"  fire_detected=1 : {forest['fire_detected'].sum()} lignes ({forest['fire_detected'].mean()*100:.2f}%)")

    top = forest.groupby("wilaya_name")["fire_detected"].sum().sort_values(ascending=False).head(15)
    print("\nTop 15 wilayas forestières par jours avec détection (2015-2026) :")
    print(top.to_string())


if __name__ == "__main__":
    main()
