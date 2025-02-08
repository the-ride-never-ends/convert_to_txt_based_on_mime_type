#!/usr/bin/env

from __future__ import annotations
from __about__ import __version__


from enum import Enum
import os
from pathlib import Path
from typing import Self


from duckdb import DuckDBPyConnection
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
    PrivateAttr,
)


from pydantic_models.playwright.proxy_launch_configs import ProxyLaunchConfigs
from pydantic_models.playwright.browser_launch_configs import BrowserLaunchConfigs
from pydantic_models.types.valid_path import ValidPath

from utils.config_parser._make_dict_keys_and_string_values_lower_case_and_strip_off_whitespace import (
    _make_dict_keys_and_string_values_lower_case_and_strip_off_whitespace
)
from utils.config_parser._check_for_whitespace_in_specific_strings import _check_for_whitespace_in_specific_strings


class Paths():
    PROJECT_ROOT: Path = Path(os.getcwd())
    INPUT_DIR: Path = PROJECT_ROOT / "input"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

# class Configs(BaseModel):
#     paths: Paths = Paths()
#     input_db: DuckDBPyConnection = None
#     output_db: DuckDBPyConnection = None
#     browser_launch_configs: BrowserLaunchConfigs = BrowserLaunchConfigs()
#     concurrency: int = Field(default=1, ge=1, le=10)

#     model_config = {
#         "arbitrary_types_allowed": True
#     }



class Configs(BaseModel):
    """
    Configs for the program. These should be considered as constants.

    Attributes:
        input_folder (str): Path to the folder containing the files to be converted.
            Defaults to 'input', the name of the input folder in the working directory.
        output_folder (str): Path to the folder where the converted files will be saved.
            Defaults to 'output', the name of the output folder in the working directory.
        max_memory (int): Maximum amount of memory in Megabytes the program can use at any one time.
            Defaults to 1024 MB.
        conversion_timeout (int): Maximum amount of time in seconds an API-bounded conversion can run before it is terminated.
            Defaults to 30 seconds.
        log_level (str): Level of logging to be used.
            Defaults to 'INFO'.
        max_connections_per_api (int): Maximum number of concurrent API connections the program can have at any one time.
            Defaults to 3.
        max_threads (int): Maximum number of threads to be used for processing the program can use at any one time.
            Defaults to 4.
        batch_size (int): Number of files to be processed in a single batch.
            Defaults to 1024.
        llm_api_key (str): API key for the LLM API.
            Defaults to 'abcde123456'.
        llm_api_url (str): URL for the LLM API.
            Defaults to 'www.example.com'.
        use_docintel (bool): Use Document Intelligence to extract text instead of offline conversion. Requires a valid Document Intelligence Endpoint.
            Defaults to False.
        docintel_endpoint (str): Document Intelligence Endpoint. Required if using Document Intelligence.
            Defaults to 'www.example2.com'.
        version (str): (CLI only) Version of the program.
            Defaults to '0.1.0'.
        help (bool): (CLI only) Show help message and exit.
            Defaults to False.
        pool_refresh_rate (int): Refresh rate in seconds for refreshing resources in the Pools.
            Defaults to 60 seconds.
        pool_health_check_rate (int): Health check rate in seconds for checking resources in the Pools.
            Defaults to 30 seconds.
        print_configs_on_startup (bool): Print the program configs to console on start-up. Sensitive values like API keys will be [REDACTED]. 
            Defaults to False.
    """
    input_folder: ValidPath = Field(default="input", min_length=1)
    output_folder: ValidPath = Field(default="output", min_length=1)
    max_memory: int = Field(default=1024, ge=128, le=16384)  # Between 128MB and 16GB
    conversion_timeout: int = Field(default=30, ge=1, le=300)  # Between 1 and 300 seconds
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    max_connections_per_api: int = Field(default=3, ge=1, le=10)
    max_threads: int = Field(default=4, ge=1, le=32)
    batch_size: int = Field(default=1024, ge=1, le=10000)
    llm_api_key: str = Field(default="abcde123456", min_length=8)
    llm_api_url: str = Field(default="http://www.example.com")
    use_docintel: bool = Field(default=False)
    docintel_endpoint: str = Field(default="http://www.example2.com")
    version: str = Field(default=__version__, pattern=r"^\d+\.\d+\.\d+$")
    pool_refresh_rate: int = Field(default=60, ge=1, le=3600)  # Between 1 second and 1 hour
    pool_health_check_rate: int = Field(default=30, ge=1, le=1800)  # Between 1 second and 30 minutes
    print_configs_on_startup: bool = Field(default=False)

    _can_use_llm = PrivateAttr(default=True)
    _can_use_docintel = PrivateAttr(default=True)

    def __init__(self, **data):

        keys_to_check = ["docintel_endpoint", "llm_api_url", "llm_api_key", "log_level"]
        data = _make_dict_keys_and_string_values_lower_case_and_strip_off_whitespace(data)
        try:
            data['log_level'] = data['log_level'].upper()
        except KeyError:
            data['log_level'] = 'INFO'
            print("Cannot find log_level in configs. Defaulting to 'INFO' ")

        _check_for_whitespace_in_specific_strings(data, keys_to_check)


        super().__init__(**data)

    @model_validator(mode='after')
    def turn_paths_into_path_objects_then_check_if_they_exist(self) -> Self:
        self.input_folder = Path(self.input_folder) if isinstance(self.input_folder, str) else self.input_folder
        self.output_folder = Path(self.output_folder) if isinstance(self.output_folder, str) else self.output_folder

        if not self.input_folder.exists():
            raise ValueError(f"Path for input_folder is invalid or does not exist: {self.input_folder}")
        if not self.output_folder.exists():
            raise ValueError(f"Path for output_folder is invalid or does not exist: {self.output_folder}")
        return self

    @model_validator(mode='after')
    def check_config_yaml_version(self) -> Self:
        if self.version != __version__:
            raise ValueError(f"Config file version {self.version} does not match the program version {__version__}. Either update/re-generate your config.yaml file, or use the command line.")
        return self

    @model_validator(mode='after')
    def check_for_mock_api_values(self) -> Self:
        if self.llm_api_url == "http://www.example.com":
            print("WARNING: The default LLM API URL is NOT a valid API endpoint. Please update the config file with an actual LLM API endpoint.")
            self._can_use_llm = False
        if self.llm_api_url == "abcde123456":
            print("WARNING: The default LLM API key is NOT a valid API key. Please update the config file with an actual LLM API key.")
            self._can_use_llm = False
        return self

    @model_validator(mode='after')
    def check_for_docintel_endpoint(self) -> Self:
        if self.use_docintel is True and self.docintel_endpoint == "http://www.example2.com":
            print("WARNING: The default Document Intelligence Endpoint is NOT a valid endpoint. Please update the config file with an actual Document Intelligence Endpoint.")
            self._can_use_docintel = False
        return self

    @property
    def can_use_llm(self) -> bool:
        return self._can_use_llm

    @property
    def can_use_docintel(self) -> bool:
        return self._can_use_docintel

