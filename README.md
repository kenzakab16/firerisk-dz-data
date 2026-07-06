# FireRisk DZ — Data Pipeline

Pipeline ETL de collecte, nettoyage et fusion de données météo et incendies pour l'Algérie, en vue de la construction d'un modèle de risque incendie de forêt et d'un tableau de bord national.

Ce dépôt contient la **phase 1** du projet FireRisk DZ : la constitution du jeu de données, couvrant **2000 → aujourd'hui**, mis à jour en continu. Le tableau de bord analytique (phase 3) est en développement dans un dépôt séparé.

## Contenu

```
data/
  processed/
    wilayas.csv                              58 wilayas : id, code officiel, nom, centroïde, superficie, flag zone forestière
    wilayas_simplified.geojson               géométries simplifiées des 58 wilayas (pour cartographie)
    weather_2000_2014.csv                    météo journalière par wilaya (Open-Meteo / ERA5), 2000-01-01 → 2014-12-31
    weather_2015_2026.csv                    météo journalière par wilaya, 2015-01-01 → aujourd'hui
    fires_daily_wilaya_2000_2026.csv         détections de feux de végétation agrégées par jour x wilaya
    ml_table_daily_wilaya_2000_2026.csv      table finale : météo + incendies, 58 wilayas, 1 ligne = 1 jour x 1 wilaya
    ml_table_forest_zone_2000_2026.csv       même table, restreinte aux 36 wilayas à couverture forestière réelle
    *.parquet                                mêmes tables en Parquet
scripts/
  01_build_wilayas.py                    géométrie + centroïdes des 58 wilayas
  02_fetch_weather.py                    météo Open-Meteo 2015-2023 (avec reprise sur erreur / rate-limit)
  03_fetch_firms.py                      détections FIRMS 2015-2023 (VIIRS SNPP + NOAA-20), toute l'Algérie
  04_spatial_join.py                     rattachement des détections FIRMS à leur wilaya (point-in-polygon)
  05_build_ml_table.py                   fusion météo + incendies en table finale (2015-2023)
  06_filter_forest_zone.py               filtre wilayas forestières + export final (2015-2023)
  07_fetch_weather_2024_2026.py          extension météo 2024 → aujourd'hui
  08_fetch_firms_2024_2026.py            extension FIRMS 2024 → aujourd'hui (SP + NRT + NOAA-21)
  09_rebuild_full_pipeline_2015_2026.py  table finale sur 2015-2026
  10_fetch_weather_2000_2014.py          extension météo en arrière, 2000-2014
  11_fetch_firms_modis_2000_2014.py      détections MODIS 2000-2014 (seul satellite existant sur cette période)
  12_rebuild_full_pipeline_2000_2026.py  reconstruit la table finale sur l'ensemble 2000-2026
  13_train_model.py                      entraîne le modèle prédictif (phase 4) et exporte model_fire_risk_v1.joblib
```

Les fichiers `data/raw/` (détections FIRMS brutes, ~2,5M lignes cumulées) **ne sont pas versionnés** pour rester sous les limites de taille de GitHub — ils sont régénérables via les scripts de collecte.

## Sources de données

| Source | Type | Période | Usage |
|---|---|---|---|
| [Open-Meteo Archive API](https://open-meteo.com/) (ERA5) | API | 1940+ | Météo journalière par wilaya |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — MODIS (2000-2014) + VIIRS SNPP/NOAA-20 (science quality) + NOAA-21 (temps quasi-réel) | API | 2000-11-01+ | Détections de feu |
| [fr33dz/Algeria-geojson](https://github.com/fr33dz/Algeria-geojson) | GeoJSON | Statique | Limites administratives des 58 wilayas |

## Méthodologie

### Météo (`02`, `07`, `10`)

Une requête par wilaya sur son centroïde, couvrant toute la période demandée en un seul appel (API Open-Meteo Archive), disponible depuis 1940. Variables : températures (max/min/moy), humidité relative, vent (vitesse/rafales/direction), précipitations, ensoleillement, rayonnement solaire, évapotranspiration (ET0 FAO), pression de surface.

Les scripts gèrent le rate-limiting (HTTP 429) avec un backoff progressif et une reprise par wilaya (checkpoint individuel), l'API Open-Meteo étant sensible au volume de données par requête sur de longues séries — certaines wilayas ont nécessité 2-3 relances.

### Incendies (`03`, `08`, `11`, jointure/filtres dans `04`/`09`/`12`)

Détections FIRMS pour toute l'Algérie (bbox complète), par tranches de 5 jours (limite de l'API `area` pour une zone de cette taille) :

- **2000-11-01 → 2014-12-31 : MODIS** (Terra + Aqua), résolution 1 km — **seul satellite existant sur cette période**. VIIRS (SNPP/NOAA-20/NOAA-21) n'existe pas avant 2012 (lancement des satellites). Un test direct de l'API confirme qu'**aucune donnée réelle n'existe avant le 2000-11-01** (mise en service opérationnelle du capteur) : janvier-octobre 2000 ne sont donc couverts par aucune détection, et ne sont **pas comblés artificiellement** — la colonne `fire_data_coverage` de la table finale le signale explicitement.
- **2015-2023** : VIIRS SNPP + NOAA-20, produit "science quality" (SP), résolution 375 m.
- **2024 → aujourd'hui** : SP tant que disponible (généralement jusqu'à J-60/90), puis bascule automatique sur le temps quasi-réel (NRT) pour les semaines les plus récentes, plus VIIRS NOAA-21 (disponible uniquement en NRT depuis janvier 2024).

⚠️ Les flux **NRT n'ont pas de colonne `type`** (contrairement à MODIS et VIIRS SP) — voir filtre 1 ci-dessous pour la conséquence méthodologique.

**Quatre filtres successifs sont appliqués**, chacun documenté car ils changent significativement le résultat :

1. **`type == 0` (feu de végétation) OU absence du champ `type` (flux NRT)** — le champ `type` de FIRMS distingue feu de végétation (0), volcan actif (1), autre source statique (2 = torchères industrielles, chaudières...), offshore (3). Sur l'Algérie entière, **78% des détections brutes disposant du champ type sont de type 2** (torchères de gaz/pétrole à Hassi Messaoud, Illizi...), sans rapport avec le risque incendie de forêt. Les détections NRT n'ayant pas ce champ, elles sont provisoirement conservées et c'est le filtre de persistance (étape 2) qui prend le relais pour exclure les torchères déjà identifiées.

2. **Filtre de persistance spatiale** — même après filtrage par type, certaines torchères restent classées type=0 par l'algorithme FIRMS (ou n'ont pas de type, cf. NRT). Un vrai feu de forêt ne réapparaît pas des centaines de fois au même endroit (~1 km) sur toute la période ; un point industriel fixe, si. On exclut les emplacements détectés plus de 15 jours distincts sur l'ensemble **2000-2026** (recalculé sur la période complète à chaque mise à jour, pour que les torchères identifiées grâce à l'historique profond soient aussi exclues du flux NRT récent).

3. **Restriction aux wilayas forestières** — malgré les deux filtres précédents, les wilayas sahariennes (Ouargla, Illizi...) concentrent encore un nombre de jours de détection largement supérieur à ce qu'un climat désertique justifierait, signe d'une contamination résiduelle par les infrastructures pétrolières denses de ces régions. La table `ml_table_forest_zone_2000_2026.csv` restreint donc l'analyse aux **36 wilayas à couverture forestière réelle** (Tell, Atlas tellien, Aurès, littoral) — c'est le périmètre pertinent pour un modèle de risque incendie de forêt. Les 22 wilayas exclues (Sahara + steppe pré-désertique) sont listées dans les scripts de reconstruction.

Après ces filtres, le classement des wilayas par nombre de jours avec détection est cohérent avec la réalité connue des feux de forêt en Algérie (Béjaïa, Skikda, Tizi Ouzou, Jijel, Médéa, Blida, Bouira en tête sur 2000-2026).

### Jointure spatiale

Rattachement de chaque détection à sa wilaya par point-in-polygon (`geopandas.sjoin`, prédicat `within`) sur les géométries simplifiées. Les géométries simplifiées se chevauchant légèrement aux frontières, un dédoublonnage garde une seule wilaya par détection (première correspondance).

## Schéma de la table finale

`ml_table_forest_zone_2000_2026.csv` / `ml_table_daily_wilaya_2000_2026.csv` — 1 ligne = 1 jour × 1 wilaya, 2000-01-01 → aujourd'hui :

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
| `area_km2`, `centroid_lat`, `centroid_lon`, `is_forest_zone` | Caractéristiques de la wilaya |
| `nb_detections`, `frp_total`, `frp_max` | Détections FIRMS du jour (0 si aucune) |
| `fire_detected` | **Variable cible** : 1 si au moins une détection de feu de végétation ce jour-là, sinon 0 |
| `fire_data_coverage` | `False` pour janvier-octobre 2000 (aucune donnée satellite réelle disponible) ; `True` sinon. À vérifier avant d'interpréter `fire_detected=0` comme "pas de feu confirmé" |

## Limites connues

- Résolution spatiale : 1 point météo par wilaya (centroïde), pas de grille — comme pour le pilote Tizi Ouzou.
- `fire_detected` est basé sur la détection satellite (MODIS 1 km puis VIIRS 375 m), pas sur les incendies officiellement déclarés — biais possible sur les très petits foyers ou en cas de forte couverture nuageuse, et résolution plus grossière avant 2015 (MODIS).
- **Aucune donnée de détection de feu n'existe avant le 2000-11-01** — vérifié directement auprès de l'API FIRMS (réponse vide), pas une limite de notre collecte. La météo, elle, est réelle et disponible dès le 2000-01-01.
- Les données les plus récentes (dernières semaines) reposent sur le flux NRT, moins retraité que le SP — possibilité de révisions mineures a posteriori quand NASA repasse ces dates en "science quality".
- Pas encore intégré : relief (altitude/pente), occupation des sols (NDVI, densité forestière), distance aux routes/villages — prévus en phase 2.
- Les wilayas sahariennes sont exclues de la table `forest_zone` mais restent disponibles dans la table complète, avec l'avertissement ci-dessus sur la contamination torchères.

## Lancer le pipeline

```bash
pip install -r requirements.txt
cd scripts
python 01_build_wilayas.py
python 02_fetch_weather.py                    # 2015-2023, ~10-15 min (rate-limit Open-Meteo)
python 03_fetch_firms.py                      # 2015-2023, ~10-15 min (~1100 requêtes FIRMS)
python 04_spatial_join.py
python 05_build_ml_table.py
python 06_filter_forest_zone.py
python 07_fetch_weather_2024_2026.py          # extension jusqu'à aujourd'hui
python 08_fetch_firms_2024_2026.py            # extension jusqu'à aujourd'hui
python 09_rebuild_full_pipeline_2015_2026.py  # table finale 2015-2026
python 10_fetch_weather_2000_2014.py          # extension en arrière, ~15-20 min (plusieurs relances possibles)
python 11_fetch_firms_modis_2000_2014.py      # ~10-15 min (~1035 requêtes MODIS)
python 12_rebuild_full_pipeline_2000_2026.py  # table finale sur l'ensemble 2000-2026
```

Pour les mises à jour suivantes (nouvelles semaines de données), relancer uniquement les scripts 07/08/09 puis 12, ou directement 12 après avoir régénéré 07/08.

## Prochaines étapes

- Phase 2 : relief (SRTM), occupation des sols (ESA WorldCover), stockage PostgreSQL/PostGIS
- Phase 3 : tableau de bord analytique (carte de risque par wilaya, corrélations météo ↔ incendies) — en cours
- Phase 4 : modèle prédictif — **v1 livrée** (`13_train_model.py`) : HistGradientBoosting entraîné sur l'ère VIIRS 2015-2023 (labels homogènes), testé sur 2024-2026 jamais vues (ROC-AUC 0,75, PR-AUC 0,33 pour un taux de base de 13%). Artefacts : `data/processed/model_fire_risk_v1.joblib` + méta JSON. Pistes v2 : variables d'assèchement antécédent (pluie cumulée 7/30 j), NDVI, relief
