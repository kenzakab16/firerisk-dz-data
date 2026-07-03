"""
Fusionne les détections FIRMS (VIIRS SNPP + NOAA-20), dédoublonne,
et rattache chaque point à sa wilaya par jointure spatiale
(point-in-polygon) avec les géométries des 58 wilayas.

Produit :
  - data/processed/fires_raw_with_wilaya.csv   (1 ligne = 1 détection)
  - data/processed/fires_daily_wilaya.csv      (1 ligne = 1 jour x 1 wilaya)
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

RAW_DIR = "../data/raw"
WILAYAS_GEOJSON = "../data/processed/wilayas_simplified.geojson"
OUT_RAW = "../data/processed/fires_raw_with_wilaya.csv"
OUT_DAILY = "../data/processed/fires_daily_wilaya.csv"


def load_source(path, satellite_label):
    df = pd.read_csv(path, parse_dates=["acq_date"])
    df["source_satellite"] = satellite_label
    return df


def main():
    print("Chargement des détections FIRMS...")
    snpp = load_source(f"{RAW_DIR}/firms_viirs_snpp_sp_algeria_2015_2023.csv", "SNPP")
    noaa20 = load_source(f"{RAW_DIR}/firms_viirs_noaa20_sp_algeria_2015_2023.csv", "NOAA20")
    fires = pd.concat([snpp, noaa20], ignore_index=True)
    print(f"  {len(fires)} détections brutes ({len(snpp)} SNPP + {len(noaa20)} NOAA20)")

    fires = fires.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    print(f"  {len(fires)} après dédoublonnage")

    print("Chargement des géométries de wilayas...")
    wilayas = gpd.read_file(WILAYAS_GEOJSON)

    print("Construction du GeoDataFrame des points...")
    geometry = [Point(xy) for xy in zip(fires["longitude"], fires["latitude"])]
    fires_gdf = gpd.GeoDataFrame(fires, geometry=geometry, crs="EPSG:4326")

    print("Jointure spatiale (point-in-polygon)...")
    fires_gdf = fires_gdf.reset_index(drop=True)
    joined = gpd.sjoin(fires_gdf, wilayas[["wilaya_id", "wilaya_code", "wilaya_name", "geometry"]],
                        how="left", predicate="within")
    # Les géométries simplifiées peuvent se chevaucher légèrement aux frontières :
    # un point peut matcher 2 polygones -> on ne garde qu'une wilaya par détection.
    n_before = len(joined)
    joined = joined[~joined.index.duplicated(keep="first")]
    if n_before != len(joined):
        print(f"  {n_before - len(joined)} doublons de jointure supprimés (chevauchement aux frontières)")
    n_matched = joined["wilaya_id"].notna().sum()
    print(f"  {n_matched}/{len(joined)} détections rattachées à une wilaya "
          f"({len(joined) - n_matched} hors zone, probablement Sahara/mer/pays voisin)")

    out = joined.drop(columns=["geometry", "index_right"])
    out.to_csv(OUT_RAW, index=False)
    print(f"Écrit {OUT_RAW}")

    # Agrégation journalière par wilaya — uniquement les feux de végétation (type=0).
    # type=2 domine le dataset brut (torchères de gaz/pétrole à Hassi Messaoud, Illizi,
    # etc.) et n'a rien à voir avec le risque incendie de forêt ; type=3 = offshore.
    matched = out[(out["wilaya_id"].notna()) & (out["type"] == 0)].copy()
    matched["wilaya_id"] = matched["wilaya_id"].astype(int)

    # Filtre de persistance spatiale : certaines torchères sont mal classées type=0
    # par l'algorithme FIRMS. Un vrai feu de forêt ne réapparaît pas des centaines de
    # fois exactement au même endroit (~1km) sur 9 ans -> on exclut les emplacements
    # récurrents (source ponctuelle fixe), seuil = 15 jours distincts sur la période.
    matched["lat_grid"] = matched["latitude"].round(2)
    matched["lon_grid"] = matched["longitude"].round(2)
    persistence = matched.groupby(["lat_grid", "lon_grid"])["acq_date"].nunique()
    PERSISTENCE_THRESHOLD = 15
    recurring_spots = persistence[persistence > PERSISTENCE_THRESHOLD].index
    n_before = len(matched)
    matched = matched.set_index(["lat_grid", "lon_grid"])
    matched = matched[~matched.index.isin(recurring_spots)].reset_index()
    print(f"  Filtre de persistance (>{PERSISTENCE_THRESHOLD}j au même endroit sur 9 ans) : "
          f"{n_before - len(matched)} détections exclues ({len(recurring_spots)} emplacements récurrents)")

    daily = matched.groupby(["wilaya_id", "wilaya_code", "acq_date"]).agg(
        nb_detections=("frp", "count"),
        frp_total=("frp", "sum"),
        frp_max=("frp", "max"),
    ).reset_index().rename(columns={"acq_date": "date"})
    daily["fire_detected"] = 1
    daily.to_csv(OUT_DAILY, index=False)
    print(f"Écrit {OUT_DAILY} — {len(daily)} lignes (jour x wilaya avec détection)")


if __name__ == "__main__":
    main()
