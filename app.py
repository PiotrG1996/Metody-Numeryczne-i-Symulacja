import runpy
import sys
from pathlib import Path

KOLOKWIUM_DIR = Path(__file__).resolve().parent / "kolokwium"
sys.path.insert(0, str(KOLOKWIUM_DIR))

runpy.run_path(str(KOLOKWIUM_DIR / "app.py"), run_name="__main__")
