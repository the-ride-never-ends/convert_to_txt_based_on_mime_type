from enum import Enum
import os
from pathlib import Path


class DefaultPaths(Path, Enum):
    PROJECT_ROOT: Path = Path(os.getcwd())
    INPUT_DIR: Path = PROJECT_ROOT / "input"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    LOG_DIR: Path = PROJECT_ROOT / "logs"