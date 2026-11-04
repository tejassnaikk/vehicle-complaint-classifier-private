"""Phase 1: download NHTSA complaint chunks into RAW_DIR and unzip them."""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR

BASE = "https://static.nhtsa.gov/odi/ffdd/cmpl/"
FILES = [
    "COMPLAINTS_RECEIVED_2020-2024.zip",
    "COMPLAINTS_RECEIVED_2025-2026.zip",
]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        zpath = RAW_DIR / name
        if not zpath.exists():
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(BASE + name, zpath)
        print(f"{name}: {zpath.stat().st_size:,} bytes")
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                print(f"  contains {info.filename} ({info.file_size:,} bytes)")
            z.extractall(RAW_DIR)


if __name__ == "__main__":
    main()
