"""
Étend les détections FIRMS au-delà de 2023 : 2024-01-01 -> aujourd'hui,
toute l'Algérie. Utilise le produit "science quality" (SP) tant qu'il
est disponible, puis bascule sur le temps quasi-réel (NRT) pour les
dates les plus récentes non encore repassées en SP. Ajoute aussi
VIIRS NOAA-21 (disponible uniquement en NRT depuis 2024-01-17).
"""
import datetime
import time
import urllib.request
import urllib.error

MAP_KEY = "3564558944d7ab736a51254db8be2620"
ALGERIA_BBOX = "-8.68,18.96,11.99,37.12"
DAY_RANGE = 5
OUT_DIR = "../data/raw"

TODAY = datetime.date.today()

# (source, date_debut, date_fin) — bornes ajustées à la disponibilité réelle des produits.
JOBS = [
    ("VIIRS_SNPP_SP", datetime.date(2024, 1, 1), datetime.date(2026, 4, 27)),
    ("VIIRS_SNPP_NRT", datetime.date(2026, 4, 28), TODAY),
    ("VIIRS_NOAA20_SP", datetime.date(2024, 1, 1), datetime.date(2026, 4, 30)),
    ("VIIRS_NOAA20_NRT", datetime.date(2026, 5, 1), TODAY),
    ("VIIRS_NOAA21_NRT", datetime.date(2024, 1, 17), TODAY),
]


def date_chunks(start, end, step_days):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=step_days)


def fetch_chunk(source, date, retries=4):
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/"
        f"{source}/{ALGERIA_BBOX}/{DAY_RANGE}/{date.isoformat()}"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt+1}/{retries} for {source} {date} after {e} (wait {wait}s)")
            time.sleep(wait)
    print(f"    ECHEC définitif pour {source} {date}")
    return None


def main():
    for source, start, end in JOBS:
        if start > end:
            continue
        out_path = f"{OUT_DIR}/firms_{source.lower()}_algeria_2024_2026.csv"
        header = None
        rows = []
        chunks = list(date_chunks(start, end, DAY_RANGE))
        print(f"=== {source} : {len(chunks)} requêtes ({start} -> {end}) ===")
        for i, d in enumerate(chunks):
            text = fetch_chunk(source, d)
            if text is None:
                continue
            lines = text.strip().split("\n")
            if len(lines) < 2:
                continue
            if header is None:
                header = lines[0]
            rows.extend(lines[1:])
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(chunks)}] {d} — {len(rows)} lignes cumulées")
            time.sleep(0.35)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write((header or "") + "\n")
            for r in rows:
                if r.strip():
                    f.write(r + "\n")
        print(f"Écrit {out_path} — {len(rows)} détections\n")


if __name__ == "__main__":
    main()
