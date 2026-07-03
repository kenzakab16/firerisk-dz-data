"""
Construit la table `wilayas` (58 wilayas) : nom, code officiel, centroïde,
superficie, géométrie simplifiée.

Source géométrie : fr33dz/Algeria-geojson (version simplifiée, 58 wilayas post-2019).
Codes officiels : liste standard des wilayas d'Algérie (ordre de création).
"""
import json
import geopandas as gpd

RAW_GEOJSON = "../data/raw/final_geodata/alg-final.geojson"
OUT_CSV = "../data/processed/wilayas.csv"
OUT_GEOJSON = "../data/processed/wilayas_simplified.geojson"

# Codes officiels des 58 wilayas (ministère de l'Intérieur), nom -> code à 2 chiffres.
OFFICIAL_CODES = {
    "Adrar": "01", "Chlef": "02", "Laghouat": "03", "Oum El Bouaghi": "04",
    "Batna": "05", "Béjaïa": "06", "Biskra": "07", "Béchar": "08",
    "Blida": "09", "Bouira": "10", "Tamanrasset": "11", "Tébessa": "12",
    "Tlemcen": "13", "Tiaret": "14", "Tizi Ouzou": "15", "Algiers": "16",
    "Djelfa": "17", "Jijel": "18", "Sétif": "19", "Saïda": "20",
    "Skikda": "21", "Sidi Bel Abbès": "22", "Annaba": "23", "Guelma": "24",
    "Constantine": "25", "Médéa": "26", "Mostaganem": "27", "M'Sila": "28",
    "Mascara": "29", "Ouargla": "30", "Oran": "31", "El Bayadh": "32",
    "Illizi": "33", "Bordj Bou Arreridj": "34", "Boumerdès": "35",
    "El Tarf": "36", "Tindouf": "37", "Tissemsilt": "38", "El Oued": "39",
    "Khenchela": "40", "Souk Ahras": "41", "Tipaza": "42", "Mila": "43",
    "Aïn Defla": "44", "Naâma": "45", "Aïn Témouchent": "46", "Ghardaia": "47",
    "Relizane": "48", "Timimoune": "49", "Bordj Badji Mokhtar": "50",
    "Ouled Djellal": "51", "Béni Abbès": "52", "In Salah": "53",
    "In Guezzam": "54", "Touggourt": "55", "Djanet": "56",
    "El M'Ghair": "57", "El Menia": "58",
}

gdf = gpd.read_file(RAW_GEOJSON)
gdf = gdf.set_crs(epsg=4326, allow_override=True)

gdf["wilaya_code"] = gdf["NAME_1"].map(OFFICIAL_CODES)
missing = gdf[gdf["wilaya_code"].isna()]
if len(missing):
    raise SystemExit(f"Noms non mappés : {missing['NAME_1'].tolist()}")

# Centroïde + superficie (projection équivalente pour un calcul correct en km2)
gdf_m = gdf.to_crs(epsg=3857)
centroids_m = gdf_m.geometry.centroid
centroids = gpd.GeoSeries(centroids_m, crs=3857).to_crs(4326)
gdf["centroid_lon"] = centroids.x.values
gdf["centroid_lat"] = centroids.y.values
gdf["area_km2"] = (gdf_m.geometry.area / 1_000_000).round(1)

out = gdf[["wilaya_code", "NAME_1", "name_ar", "name_ber", "area_km2", "centroid_lat", "centroid_lon"]].copy()
out.columns = ["wilaya_code", "wilaya_name", "wilaya_name_ar", "wilaya_name_ber", "area_km2", "centroid_lat", "centroid_lon"]
out = out.sort_values("wilaya_code").reset_index(drop=True)
out["wilaya_id"] = range(1, len(out) + 1)
out = out[["wilaya_id", "wilaya_code", "wilaya_name", "wilaya_name_ar", "wilaya_name_ber",
           "area_km2", "centroid_lat", "centroid_lon"]]

out.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"Écrit {OUT_CSV} — {len(out)} wilayas")

# Géométrie simplifiée pour la carte (tolérance ~500m)
gdf_simplified = gdf.copy()
gdf_simplified["geometry"] = gdf_simplified.geometry.simplify(0.005, preserve_topology=True)
gdf_simplified = gdf_simplified.merge(out[["wilaya_code", "wilaya_id"]], on="wilaya_code")
gdf_simplified = gdf_simplified[["wilaya_id", "wilaya_code", "NAME_1", "geometry"]]
gdf_simplified.columns = ["wilaya_id", "wilaya_code", "wilaya_name", "geometry"]
gdf_simplified.to_file(OUT_GEOJSON, driver="GeoJSON")
print(f"Écrit {OUT_GEOJSON}")

print(out[["wilaya_id", "wilaya_code", "area_km2", "centroid_lat", "centroid_lon"]].head(10).to_string())
