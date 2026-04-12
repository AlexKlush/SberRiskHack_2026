"""Helper to switch between test datasets.

Usage:
    python switch_dataset.py pet      # switch to pet_supplies dataset
    python switch_dataset.py ds1      # switch to dataset 1 (e-commerce)
    python switch_dataset.py ds2      # switch to dataset 2 (client_data)
"""
import sys
import shutil
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
BACKUP_DIR = BASE / "data_backups"

DATASETS = {
    "pet": BASE / "data_pet",
    # Add more as needed — just create data_<name>/ folders
}


def backup_current():
    """Save current data/ as a numbered backup if it has files."""
    csvs = list(DATA.glob("*.csv"))
    if not csvs:
        return
    # Check if it matches a known dataset
    BACKUP_DIR.mkdir(exist_ok=True)
    idx = len(list(BACKUP_DIR.glob("backup_*")))
    dest = BACKUP_DIR / f"backup_{idx}"
    shutil.copytree(DATA, dest)
    print(f"  Backed up current data/ -> {dest}")


def switch(name):
    src = DATASETS.get(name)
    if src is None:
        # Try data_<name> folder
        src = BASE / f"data_{name}"
    if not src.exists():
        print(f"Error: dataset folder '{src}' not found")
        print(f"Available: {list(DATASETS.keys())}")
        sys.exit(1)

    # Clear data/
    for f in DATA.glob("*"):
        if f.is_file():
            f.unlink()

    # Copy new files
    for f in src.glob("*"):
        if f.is_file():
            shutil.copy2(f, DATA / f.name)
    print(f"  Switched data/ to '{name}' ({len(list(DATA.glob('*.csv')))} files)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    name = sys.argv[1]
    backup_current()
    switch(name)
    print("  Done! Run: python run.py")
