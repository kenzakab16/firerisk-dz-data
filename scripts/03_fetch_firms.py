"""
Récupère toutes les détections de feu NASA FIRMS (VIIRS SNPP + NOAA-20,
produit science quality) sur l'Algérie entière, 2015-01-01 -> 2023-12-31.

L'API area renvoie au maximum 10 jours par requête -> on boucle par
tranches de 10 jours. NOAA-20 SP n'est disponible qu'à partir du
2018-04-01 (limite du produit science quality).
"""
import datetime
import time
import urllib.request
import urllib.error

MAP_KEY = "3564558944d7ab736a51254db8be2620"
ALGERIA_BBOX = "-8.68,18.96,11.99,37.12"
DAY_RANGE = 5
OUT_DIR = "../data/raw"

SOURCES = {
    "VIIRS_SNPP_SP": (datetime.date(2015, 1, 1), datetime.date(2023, 12, 31)),
    "VIIRS_NOAA20_SP": (datetime.date(2018, 4, 1), datetime.date(2023, 12, 31)),
}


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
    for source, (start, end) in SOURCES.items():
        out_path = f"{OUT_DIR}/firms_{source.lower()}_algeria_2015_2023.csv"
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
