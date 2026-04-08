from pathlib import Path

# Project root: ITALIAN-PARLIAMENT-ANALYSIS/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUTS_DIR = BASE_DIR / 'outputs'
