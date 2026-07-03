"""
Ajoute un flag `is_forest_zone` à la table wilayas et produit la version
filtrée de la table ML restreinte aux wilayas avec couverture forestière
réelle (nord du pays / Tell / Atlas tellien / Aurès).

Les wilayas sahariennes et pré-désertiques sont exclues : elles n'ont
pas de risque incendie de forêt pertinent, et leur signal FIRMS est de
toute façon dominé par les torchères de gaz/pétrole (Hassi Messaoud,
Illizi...), un bruit qu'un simple filtre type/persistance ne suffit
pas à éliminer complètement.
"""
import pandas as pd

WILAYAS_CSV = "../data/processed/wilayas.csv"
ML_TABLE_CSV = "../data/processed/ml_table_daily_wilaya_2015_2023.csv"
OUT_WILAYAS = "../data/processed/wilayas.csv"
OUT_ML_FILTERED = "../data/processed/ml_table_forest_zone_2015_2023.csv"
OUT_ML_FILTERED_PARQUET = "../data/processed/ml_table_forest_zone_2015_2023.parquet"

# Wilayas exclues : Sahara + pré-désert (steppe aride), sans couverture
# forestière significative.
NON_FOREST_WILAYAS = {
    "Adrar", "Béchar", "Tamanrasset", "Ouargla", "Illizi", "Tindouf",
    "El Oued", "Ghardaia", "El Bayadh", "Béni Abbès", "In Salah",
    "In Guezzam", "Timimoune", "Bordj Badji Mokhtar", "Touggourt",
    "Djanet", "El M'Ghair", "El Menia", "Ouled Djellal", "Naâma",
    "Laghouat", "Biskra",
}


def main():
    wilayas = pd.read_csv(WILAYAS_CSV)
    wilayas["is_forest_zone"] = ~wilayas["wilaya_name"].isin(NON_FOREST_WILAYAS)
    wilayas.to_csv(OUT_WILAYAS, index=False)
    n_forest = wilayas["is_forest_zone"].sum()
    print(f"wilayas.csv mis à jour — {n_forest}/{len(wilayas)} wilayas en zone forestière")

    ml = pd.read_csv(ML_TABLE_CSV, parse_dates=["date"])
    forest_ids = set(wilayas.loc[wilayas["is_forest_zone"], "wilaya_id"])
    ml_forest = ml[ml["wilaya_id"].isin(forest_ids)].copy()
    ml_forest.to_csv(OUT_ML_FILTERED, index=False)
    ml_forest.to_parquet(OUT_ML_FILTERED_PARQUET, index=False)

    print(f"Écrit {OUT_ML_FILTERED} — {len(ml_forest)} lignes "
          f"({ml_forest['wilaya_id'].nunique()} wilayas x {ml_forest['date'].nunique()} jours)")
    print(f"  fire_detected=1 : {ml_forest['fire_detected'].sum()} lignes "
          f"({ml_forest['fire_detected'].mean()*100:.2f}%)")
    print()
    top = ml_forest.groupby("wilaya_name")["fire_detected"].sum().sort_values(ascending=False).head(15)
    print("Top 15 wilayas forestières par jours avec détection (2015-2023) :")
    print(top.to_string())


if __name__ == "__main__":
    main()
