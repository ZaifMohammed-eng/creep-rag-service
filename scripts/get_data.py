"""Download SQuAD v1.1 dev (CC BY-SA 4.0, Rajpurkar et al. 2016) into data/raw/."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "dev-v1.1.json"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"{OUT} already present")
    else:
        OUT.write_bytes(urlopen(URL, timeout=60).read())
        print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
