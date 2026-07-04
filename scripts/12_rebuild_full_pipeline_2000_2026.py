"""
Reconstruit la chaîne complète sur 2000-2026 (données réelles
uniquement, aucune valeur inventée) :

  1. Fusionne MODIS (2000-11-01 -> 2014-12-31, seul satellite existant
     sur cette période) + VIIRS SNPP/NOAA-20/NOAA-21 (2015+)
  2. Jointure spatiale vers les 58 wilayas
  3. Filtre type=0 (feu de végétation) OU type manquant (flux NRT)
  4. Filtre de persistance spatiale (torchères), recalculé sur 2000-2026
  5. Fusion avec la météo complète (2000-2014 + 2015-2026) -> table ML
  6. Filtre zone forestière (36 wilayas)

Aucune couverture satellite n'existe avant le 2000-11-01 (mise en
service du capteur MODIS/Terra) : ces mois (janvier-octobre 2000) ne
sont pas comblés artificiellement, ils restent absents du dataset de
détections (la météo, elle, est réelle et disponible dès 2000-01-01).
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

RAW_DIR = "../data/raw"
WILAYAS_GEOJSON = "../data/processed/wilayas_simplified.geojson"
WILAYAS_CSV = "../data/processed/wilayas.csv"
WEATHER_2000_2014 = "../data/processed/weather_2000_2014.csv"
WEATHER_2015_2026 = "../data/processed/weather_2015_2026.csv"

OUT_FIRES_DAILY = "../data/processed/fires_daily_wilaya_2000_2026.csv"
OUT_ML = "../data/processed/ml_table_daily_wilaya_2000_2026.csv"
OUT_ML_PARQUET = "../data/processed/ml_table_daily_wilaya_2000_2026.parquet"
OUT_ML_FOREST = "../data/processed/ml_table_forest_zone_2000_2026.csv"
OUT_ML_FOREST_PARQUET = "../data/processed/ml_table_forest_zone_2000_2026.parquet"

COMMON_COLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite",
               "instrument", "confidence", "version", "frp", "daynight", "type"]

SOURCES = [
    ("firms_modis_sp_algeria_2000_2014.csv", "MODIS"),
    ("firms_viirs_snpp_sp_algeria_2015_2023.csv", "VIIRS"),
    ("firms_viirs_noaa20_sp_algeria_2015_2023.csv", "VIIRS"),
    ("firms_viirs_snpp_sp_algeria_2024_2026.csv", "VIIRS"),
    ("firms_viirs_snpp_nrt_algeria_2024_2026.csv", "VIIRS"),
    ("firms_viirs_noaa20_sp_algeria_2024_2026.csv", "VIIRS"),
    ("firms_viirs_noaa20_nrt_algeria_2024_2026.csv", "VIIRS"),
    ("firms_viirs_noaa21_nrt_algeria_2024_2026.csv", "VIIRS"),
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
    print("=== 1. Fusion des sources FIRMS (MODIS 2000-2014 + VIIRS 2015-2026) ===")
    frames = []
    for fname, product in SOURCES:
        path = f"{RAW_DIR}/{fname}"
        try:
            df = pd.read_csv(path, parse_dates=["acq_date"])
        except FileNotFoundError:
            print(f"  (absent, ignoré) {fname}")
            continue
        if "type" not in df.columns:
            df["type"] = pd.NA
        df = df[COMMON_COLS].copy()
        df["product"] = product
        frames.append(df)
        print(f"  {fname}: {len(df)} lignes ({df['acq_date'].min().date()} -> {df['acq_date'].max().date()})")

    fires = pd.concat(frames, ignore_index=True)
    fires = fires.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    print(f"Total après dédoublonnage : {len(fires)} ({fires['acq_date'].min().date()} -> {fires['acq_date'].max().date()})")

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

    print("\n=== 3. Filtre type (végétation ou NRT sans info type) ===")
    matched = out[out["wilaya_id"].notna()].copy()
    matched["wilaya_id"] = matched["wilaya_id"].astype(int)
    is_vegetation_or_unknown = (matched["type"] == 0) | (matched["type"].isna())
    matched = matched[is_vegetation_or_unknown].copy()
    print(f"  {len(matched)} détections conservées")

    print("\n=== 4. Filtre de persistance spatiale (torchères), recalculé sur 2000-2026 ===")
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

    print("\n=== 5. Fusion météo (2000-2014 + 2015-2026) + incendies ===")
    weather = pd.concat([
        pd.read_csv(WEATHER_2000_2014, parse_dates=["date"]),
        pd.read_csv(WEATHER_2015_2026, parse_dates=["date"]),
    ], ignore_index=True).drop_duplicates(subset=["wilaya_id", "date"]).sort_values(["wilaya_id", "date"])

    fires_small = daily[["wilaya_id", "date", "nb_detections", "frp_total", "frp_max", "fire_detected"]].copy()
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

    # Aucune couverture satellite réelle avant le 2000-11-01 : on le marque
    # explicitement plutôt que de laisser croire que fire_detected=0 signifie
    # "pas de feu confirmé" pour cette fenêtre.
    merged["fire_data_coverage"] = merged["date"] >= "2000-11-01"

    cols_order = [
        "date", "wilaya_id", "wilaya_code", "wilaya_name",
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
        "precipitation_sum", "rain_sum",
        "sunshine_duration", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "surface_pressure_mean",
        "area_km2", "centroid_lat", "centroid_lon", "is_forest_zone",
        "nb_detections", "frp_total", "frp_max", "fire_detected", "fire_data_coverage",
    ]
    merged = merged[cols_order].sort_values(["wilaya_id", "date"]).reset_index(drop=True)
    merged.to_csv(OUT_ML, index=False)
    merged.to_parquet(OUT_ML_PARQUET, index=False)
    print(f"Écrit {OUT_ML} — {len(merged)} lignes "
          f"({merged['wilaya_id'].nunique()} wilayas x {merged['date'].nunique()} jours, "
          f"{merged['date'].min().date()} -> {merged['date'].max().date()})")
    print(f"  dont {(~merged['fire_data_coverage']).sum()} lignes sans couverture satellite "
          f"(jan-oct 2000, {(~merged['fire_data_coverage']).sum() // merged['wilaya_id'].nunique()} jours x 58 wilayas)")

    print("\n=== 6. Filtre zone forestière ===")
    forest = merged[merged["is_forest_zone"]].copy()
    forest.to_csv(OUT_ML_FOREST, index=False)
    forest.to_parquet(OUT_ML_FOREST_PARQUET, index=False)
    print(f"Écrit {OUT_ML_FOREST} — {len(forest)} lignes ({forest['wilaya_id'].nunique()} wilayas)")
    covered = forest[forest["fire_data_coverage"]]
    print(f"  fire_detected=1 : {covered['fire_detected'].sum()} lignes sur période couverte "
          f"({covered['fire_detected'].mean()*100:.2f}%)")

    top = covered.groupby("wilaya_name")["fire_detected"].sum().sort_values(ascending=False).head(15)
    print("\nTop 15 wilayas forestières par jours avec détection (2000-2026, période couverte) :")
    print(top.to_string())


if __name__ == "__main__":
    main()
