"""
Détections de feu réelles pour 2000-2014 : NASA FIRMS ne propose pas de
VIIRS avant 2012 (satellites SNPP/NOAA-20 lancés après). Le seul
produit couvrant cette période est MODIS (Terra+Aqua), disponible en
"science quality" depuis le 2000-11-01 (mise en service opérationnelle
du capteur) -- confirmé par un test direct de l'API (réponse vide
avant cette date, aucune donnée n'existe pour janvier-octobre 2000).

Toute la période 2000-01-01 -> 2000-10-31 n'a donc AUCUNE couverture
satellite réelle : on ne la simule pas, elle reste absente du dataset.
"""
import datetime
import time
import urllib.request
import urllib.error

MAP_KEY = "3564558944d7ab736a51254db8be2620"
ALGERIA_BBOX = "-8.68,18.96,11.99,37.12"
DAY_RANGE = 5
OUT_DIR = "../data/raw"

SOURCE = "MODIS_SP"
START_DATE = datetime.date(2000, 11, 1)  # première date réelle disponible (vérifiée via l'API)
END_DATE = datetime.date(2014, 12, 31)


def date_chunks(start, end, step_days):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=step_days)


def fetch_chunk(date, retries=4):
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/"
        f"{SOURCE}/{ALGERIA_BBOX}/{DAY_RANGE}/{date.isoformat()}"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt+1}/{retries} for {date} after {e} (wait {wait}s)")
            time.sleep(wait)
    print(f"    ECHEC définitif pour {date}")
    return None


def main():
    out_path = f"{OUT_DIR}/firms_modis_sp_algeria_2000_2014.csv"
    header = None
    rows = []
    chunks = list(date_chunks(START_DATE, END_DATE, DAY_RANGE))
    print(f"=== {SOURCE} : {len(chunks)} requêtes ({START_DATE} -> {END_DATE}) ===")
    for i, d in enumerate(chunks):
        text = fetch_chunk(d)
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
    print(f"Écrit {out_path} — {len(rows)} détections")


if __name__ == "__main__":
    main()
