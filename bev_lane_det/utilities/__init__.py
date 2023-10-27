from pathlib import Path
from os import path
import logging

logger_level = logging.INFO

# __file__ = "./__init__.py"
THIS_DIR = Path(path.dirname(path.abspath(__file__)))
PROJECT_DIR = THIS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# logger
logging.getLogger('bev_lane_det').addHandler(logging.NullHandler())
