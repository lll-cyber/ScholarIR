import logging
import sys
from datetime import datetime
from pathlib import Path

# ScholarIR/logs (…/ScholarIR/src/scholar_ir/vendor_spar/log.py → parents[3])
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
log_dir = _PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

current_date = datetime.now().strftime("%Y%m%d")
logging_file_path = log_dir / f"scholar_ir_{current_date}.log"

handlers = [
    logging.FileHandler(logging_file_path),
    logging.StreamHandler(sys.stdout),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
    force=True,
)

logger = logging.getLogger("scholar_ir")
