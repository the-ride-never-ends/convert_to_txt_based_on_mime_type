from enum import Enum
import os
from pathlib import Path


from duckdb import DuckDBPyConnection
from pydantic import BaseModel, Field


from .playwright.proxy_launch_configs import ProxyLaunchConfigs
from .playwright.browser_launch_configs import BrowserLaunchConfigs


_PROJECT_ROOT = Path(os.getcwd())

class Paths(Path, Enum):
    INPUT_DIR: Path = _PROJECT_ROOT / "input"
    OUTPUT_DIR: Path = _PROJECT_ROOT / "output"
    LOG_DIR: Path = _PROJECT_ROOT / "logs"

class Configs(BaseModel):
    paths: Paths = Paths()
    input_db: DuckDBPyConnection = None
    output_db: DuckDBPyConnection = None
    browser_launch_configs: BrowserLaunchConfigs = BrowserLaunchConfigs()
    concurrency: int = Field(default=1, ge=1, le=10)








