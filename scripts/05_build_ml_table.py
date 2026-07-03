"""
Construit la table finale prête pour le machine learning :
1 ligne = 1 jour x 1 wilaya, sur 2015-01-01 -> 2023-12-31 (58 wilayas).

Fusionne weather_2015_2023.csv (toujours 1 ligne/jour/wilaya) avec
fires_daily_wilaya.csv (seulement les jours avec détection) :
les jours sans détection reçoivent fire_detected = 0 et les compteurs à 0.
"""
import pandas as pd

WEATHER_CSV = "../data/processed/weather_2015_2023.csv"
FIRES_CSV = "../data/processed/fires_daily_wilaya.csv"
WILAYAS_CSV = "../data/processed/wilayas.csv"
OUT_CSV = "../data/processed/ml_table_daily_wilaya_2015_2023.csv"
OUT_PARQUET = "../data/processed/ml_table_daily_wilaya_2015_2023.parquet"


def main():
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    fires = pd.read_csv(FIRES_CSV, parse_dates=["date"])
    wilayas = pd.read_csv(WILAYAS_CSV)

    fires_small = fires[["wilaya_id", "date", "nb_detections", "frp_total", "frp_max", "fire_detected"]]

    merged = weather.merge(fires_small, on=["wilaya_id", "date"], how="left")
    merged["nb_detections"] = merged["nb_detections"].fillna(0).astype(int)
    merged["frp_total"] = merged["frp_total"].fillna(0.0).round(2)
    merged["frp_max"] = merged["frp_max"].fillna(0.0).round(2)
    merged["fire_detected"] = merged["fire_detected"].fillna(0).astype(int)

    merged = merged.merge(
        wilayas[["wilaya_id", "wilaya_name", "area_km2", "centroid_lat", "centroid_lon"]],
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
        "area_km2", "centroid_lat", "centroid_lon",
        "nb_detections", "frp_total", "frp_max", "fire_detected",
    ]
    merged = merged[cols_order].sort_values(["wilaya_id", "date"]).reset_index(drop=True)

    merged.to_csv(OUT_CSV, index=False)
    try:
        merged.to_parquet(OUT_PARQUET, index=False)
        parquet_msg = f" + {OUT_PARQUET}"
    except ImportError:
        parquet_msg = " (parquet ignoré, pyarrow non installé)"

    print(f"Écrit {OUT_CSV}{parquet_msg}")
    print(f"  {len(merged)} lignes = {merged['wilaya_id'].nunique()} wilayas x "
          f"{merged['date'].nunique()} jours")
    print(f"  fire_detected=1 : {merged['fire_detected'].sum()} lignes "
          f"({merged['fire_detected'].mean()*100:.2f}% des lignes)")
    print()
    top = merged.groupby("wilaya_name")["fire_detected"].sum().sort_values(ascending=False).head(10)
    print("Top 10 wilayas par nombre de jours avec détection (2015-2023) :")
    print(top.to_string())


if __name__ == "__main__":
    main()
