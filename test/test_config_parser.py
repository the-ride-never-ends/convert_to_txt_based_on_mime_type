#!/usr/bin/env python3 -m pytest

import argparse
import json
from unittest.mock import patch


import pytest
import yaml


from config_parser.config_parser import ConfigParser, Configs


@pytest.fixture
def valid_config_dict():
    """Valid configuration dictionary with all default values"""
    dict_ = {
        "input_folder": "input",
        "output_folder": "output",
        "max_memory": 1024,
        "conversion_timeout": 30,
        "log_level": "INFO",
        "max_connections_per_api": 3,
        "max_threads": 4,
        "batch_size": 1024,
        "llm_api_key": "abcde12345",
        "llm_api_url": "www.example.com",
        "use_docintel": False,
        "docintel_endpoint": "www.example2.com",
        "version": "0.1.0",
        "help": False,
        "pool_refresh_rate": 60,
        "pool_health_check_rate": 30
    }
    return Configs(**dict_)


@pytest.fixture
def minimal_valid_config_dict():
    """Minimal valid configuration with only required fields"""
    dict_ = {
        "input_folder": "abcde12345",
        "output_folder": "www.example.com"
    }
    return Configs(**dict_)


@pytest.fixture
def invalid_config_dicts():
    """Dictionary of various invalid configurations for testing"""
    return {
        "missing_required": {
            "input_folder": "input",
            "output_folder": "output"
        },
        "invalid_types": {
            "llm_api_key": 6.14,
            "llm_api_url": False,
            "max_memory": "not_an_integer",
            "max_threads": 3.14
        },
        "out_of_range": {
            "llm_api_key": "abcde12345&&&&&#####$$$$$````",
            "llm_api_url": "wwwwwwwwwwwwww.example.com",
            "max_connections_per_api": -1,
            "max_threads": 0
        },
        "invalid_pairwise": {
            "llm_api_key": "abcde12345",
            "llm_api_url": "",  # Empty URL
            "use_docintel": True,
            "docintel_endpoint": ""  # Empty endpoint despite use_docintel=True
        }
    }


@pytest.fixture
def valid_config_file(tmp_path):
    """Creates a valid configs file for testing"""
    config_path = tmp_path / "configs.yaml"
    config_content = {
        "input_folder": "test_input",
        "output_folder": "test_output",
        "max_memory": 2048,
        "conversion_timeout": 60,
        "log_level": "DEBUG",
        "max_connections_per_api": 5,
        "max_threads": 8,
        "batch_size": 512,
        "llm_api_key": "test_key_123",
        "llm_api_url": "https://test.example.com",
        "use_docintel": True,
        "docintel_endpoint": "https://test.example2.com",
        "pool_refresh_rate": 30,
        "pool_health_check_rate": 15
    }
    config_path.write_text(yaml.dump(config_content))
    return config_path


@pytest.fixture
def command_line_args():
    """Dictionary of various command line argument combinations"""
    return {
        "minimal": [
            "--input-folder", "cli_input",
            "--output-folder", "cli_output",
        ],
        "full": [
            "--input-folder", "cli_input",
            "--output-folder", "cli_output",
            "--max-memory", "4096",
            "--conversion-timeout", "90",
            "--log-level", "WARNING",
            "--max-connections-per-api", "10",
            "--max-threads", "16",
            "--batch-size", "256",
            "--llm-api-key", "cli_test_key",
            "--llm-api-url", "https://cli.example.com",
            "--use-docintel",
            "--docintel-endpoint", "https://cli.example2.com",
            "--pool-refresh-rate", "45",
            "--pool-health-check-rate", "20"
        ],
        "help": ["--help"],
        "version": ["--version"]
    }


@pytest.fixture
def mock_environment_vars():
    """Dictionary of environment variables for testing"""
    return {
        "LLM_API_KEY": "env_test_key",
        "LLM_API_URL": "https://env.example.com",
        "MAX_MEMORY": "8192",
        "LOG_LEVEL": "ERROR"
    }


##### TEST FUNCTIONS #####


def test_load_valid_config_file(valid_config_file):
    parser = ConfigParser()
    parser.configs_file = valid_config_file
    configs = parser.load_and_parse_configs_file()
    assert configs is not None


def test_parse_minimal_command_line(command_line_args):
    parser = ConfigParser()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--llm-api-key', required=True)
    arg_parser.add_argument('--llm-api-url', required=True)
    args = arg_parser.parse_args(command_line_args['minimal'])

    with patch.object(parser, 'parse_command_line', return_value=vars(args)):
        configs = parser.parse_command_line(args)
        assert configs is not None
        assert configs['llm_api_key'] == 'cli_test_key'
        assert configs['llm_api_url'] == 'https://cli.example.com'


def test_save_config(tmp_path, valid_config_dict):
    parser = ConfigParser()
    parser.configs_file_path = tmp_path / "configs.yaml"
    parser.save_current_config_settings_to_configs_file(valid_config_dict)
    assert parser.configs_file_path.exists()