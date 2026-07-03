# FireRisk DZ — Data Pipeline

Pipeline ETL de collecte, nettoyage et fusion de données météo et incendies pour l'Algérie, en vue de la construction d'un modèle de risque incendie de forêt et d'un tableau de bord national.

Ce dépôt contient la **phase 1** du projet FireRisk DZ : la constitution du jeu de données. Le tableau de bord analytique et le modèle prédictif viendront dans des dépôts séparés une fois cette base validée.

## Contenu

```
data/
  processed/
    wilayas.csv                              58 wilayas : id, code officiel, nom, centroïde, superficie, flag zone forestière
    wilayas_simplified.geojson               géométries simplifiées des 58 wilayas (pour cartographie)
    weather_2015_2023.csv                    météo journalière par wilaya (Open-Meteo / ERA5), 2015-01-01 → 2023-12-31
    fires_daily_wilaya.csv                   détections de feux de végétation agrégées par jour x wilaya
    ml_table_daily_wilaya_2015_2023.csv      table finale : météo + incendies, 58 wilayas, 1 ligne = 1 jour x 1 wilaya
    ml_table_forest_zone_2015_2023.csv       même table, restreinte aux 36 wilayas à couverture forestière réelle
    *.parquet                                mêmes tables en Parquet
scripts/
  01_build_wilayas.py       géométrie + centroïdes des 58 wilayas
  02_fetch_weather.py       météo Open-Meteo (avec reprise sur erreur / rate-limit)
  03_fetch_firms.py         détections FIRMS (VIIRS SNPP + NOAA-20), toute l'Algérie
  04_spatial_join.py        rattachement des détections FIRMS à leur wilaya (point-in-polygon)
  05_build_ml_table.py      fusion météo + incendies en table finale
  06_filter_forest_zone.py  filtre wilayas forestières + export final
```

Les fichiers `data/raw/` (détections FIRMS brutes, ~1,6M lignes, ~160 Mo) **ne sont pas versionnés** pour rester sous les limites de taille de GitHub — ils sont régénérables via les scripts 02/03.

## Sources de données

| Source | Type | Période | Usage |
|---|---|---|---|
| [Open-Meteo Archive API](https://open-meteo.com/) (ERA5) | API | 1940+ | Météo journalière par wilaya |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — VIIRS SNPP + NOAA-20 | API | 2015 (SNPP) / 2018 (NOAA-20)+ | Détections de feu (science quality) |
| [fr33dz/Algeria-geojson](https://github.com/fr33dz/Algeria-geojson) | GeoJSON | Statique | Limites administratives des 58 wilayas |

## Méthodologie

### Météo (`02_fetch_weather.py`)

Une requête par wilaya sur son centroïde, couvrant toute la période 2015-2023 en un seul appel (API Open-Meteo Archive). Variables : températures (max/min/moy), humidité relative, vent (vitesse/rafales/direction), précipitations, ensoleillement, rayonnement solaire, évapotranspiration (ET0 FAO), pression de surface.

Le script gère le rate-limiting (HTTP 429) avec un backoff progressif et une reprise par wilaya (checkpoint individuel), l'API Open-Meteo étant sensible au volume de données par requête sur de longues séries.

### Incendies (`03_fetch_firms.py` + `04_spatial_join.py`)

Détections FIRMS pour toute l'Algérie (bbox complète), par tranches de 5 jours (limite de l'API `area` pour une zone de cette taille), pour VIIRS SNPP (2015+) et NOAA-20 (2018+, NOAA-21 pas encore disponible en qualité "science" sur la période).

**Trois filtres successifs sont appliqués**, chacun documenté car ils changent significativement le résultat :

1. **`type == 0` (feu de végétation)** — le champ `type` de FIRMS distingue feu de végétation (0), volcan actif (1), autre source statique (2 = torchères industrielles, chaudières...), offshore (3). Sur l'Algérie entière, **78% des détections brutes sont de type 2** (torchères de gaz/pétrole à Hassi Messaoud, Illizi...), sans rapport avec le risque incendie de forêt.

2. **Filtre de persistance spatiale** — même après filtrage par type, certaines torchères restent classées type=0 par l'algorithme FIRMS. Un vrai feu de forêt ne réapparaît pas des centaines de fois au même endroit (~1 km) sur 9 ans ; un point industriel fixe, si. On exclut les emplacements détectés plus de 15 jours distincts sur la période.

3. **Restriction aux wilayas forestières** — malgré les deux filtres précédents, les wilayas sahariennes (Ouargla, Illizi...) concentrent encore un nombre de jours de détection largement supérieur à ce qu'un climat désertique justifierait, signe d'une contamination résiduelle par les infrastructures pétrolières denses de ces régions. La table `ml_table_forest_zone_2015_2023.csv` restreint donc l'analyse aux **36 wilayas à couverture forestière réelle** (Tell, Atlas tellien, Aurès, littoral) — c'est le périmètre pertinent pour un modèle de risque incendie de forêt. Les 22 wilayas exclues (Sahara + steppe pré-désertique) sont listées dans `06_filter_forest_zone.py`.

Après ces filtres, le classement des wilayas par nombre de jours avec détection redevient cohérent avec la réalité connue des feux de forêt en Algérie (Skikda, Béjaïa, Tizi Ouzou, Blida, Jijel, Boumerdès, Sétif en tête).

### Jointure spatiale

Rattachement de chaque détection à sa wilaya par point-in-polygon (`geopandas.sjoin`, prédicat `within`) sur les géométries simplifiées. Les géométries simplifiées se chevauchant légèrement aux frontières, un dédoublonnage garde une seule wilaya par détection (première correspondance).

## Schéma de la table finale

`ml_table_forest_zone_2015_2023.csv` / `ml_table_daily_wilaya_2015_2023.csv` — 1 ligne = 1 jour × 1 wilaya :

| Colonne | Description |
|---|---|
| `date` | Date (AAAA-MM-JJ) |
| `wilaya_id`, `wilaya_code`, `wilaya_name` | Identifiants wilaya |
| `temperature_2m_max/min/mean` | Température (°C) |
| `relative_humidity_2m_mean` | Humidité relative moyenne (%) |
| `wind_speed_10m_max`, `wind_gusts_10m_max`, `wind_direction_10m_dominant` | Vent |
| `precipitation_sum`, `rain_sum` | Précipitations (mm) |
| `sunshine_duration`, `shortwave_radiation_sum` | Ensoleillement / rayonnement |
| `et0_fao_evapotranspiration` | Évapotranspiration de référence (mm) |
| `surface_pressure_mean` | Pression de surface |
| `area_km2`, `centroid_lat`, `centroid_lon` | Caractéristiques de la wilaya |
| `nb_detections`, `frp_total`, `frp_max` | Détections FIRMS du jour (0 si aucune) |
| `fire_detected` | **Variable cible** : 1 si au moins une détection de feu de végétation ce jour-là, sinon 0 |

## Limites connues

- Résolution spatiale : 1 point météo par wilaya (centroïde), pas de grille — comme pour le pilote Tizi Ouzou.
- `fire_detected` est basé sur la détection satellite (VIIRS, résolution 375 m), pas sur les incendies officiellement déclarés — biais possible sur les très petits foyers ou en cas de forte couverture nuageuse.
- Pas encore intégré : relief (altitude/pente), occupation des sols (NDVI, densité forestière), distance aux routes/villages — prévus en phase 2.
- Les wilayas sahariennes sont exclues de la table `forest_zone` mais restent disponibles dans la table complète, avec l'avertissement ci-dessus sur la contamination torchères.

## Lancer le pipeline

```bash
pip install -r requirements.txt
cd scripts
python 01_build_wilayas.py
python 02_fetch_weather.py       # ~10-15 min (rate-limit Open-Meteo)
python 03_fetch_firms.py         # ~10-15 min (~1100 requêtes FIRMS)
python 04_spatial_join.py
python 05_build_ml_table.py
python 06_filter_forest_zone.py
```

## Prochaines étapes

- Phase 2 : relief (SRTM), occupation des sols (ESA WorldCover), stockage PostgreSQL/PostGIS
- Phase 3 : tableau de bord analytique (carte de risque par wilaya, corrélations météo ↔ incendies)
- Phase 4 : modèle prédictif (Random Forest / XGBoost / LightGBM) sur `fire_detected`
