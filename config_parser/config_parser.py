
import argparse
from argparse import Namespace
from pathlib import Path
import sys
from typing import Optional


from pydantic import BaseModel
import yaml


#from pydantic_models.configs import Configs
class Configs(BaseModel):
    input_folder: str = "input"
    output_folder: str = "output"
    max_memory: int = 1024
    conversion_timeout: int = 30
    log_level: str = "INFO"
    max_connections_per_api: int = 3
    max_threads: int = 4
    batch_size: int = 1024
    llm_api_key: str = "abcde123456"
    llm_api_url: str = "www.example.com"
    use_docintel: bool = False
    docintel_endpoint: str = "www.example2.com"
    version: str = "1.0.0"
    pool_refresh_rate: int = 60
    pool_health_check_rate: int = 30


class ConfigParser:
    """
    Load, save, and parse external commands, from a config file, command line, or both.
    NOTE: The configuration file is hard-coded to be named 'config.yaml'. 
        If renamed or deleted, the program will attempt to create another using hard-coded default values.
        This includes mock-values for API keys and URLs.

    External Commands:
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

    Attributes:
    - configs_file_path: Optional[str]: The path to the config file.
    - config: BaseModel: A pydantic model containing the parsed config settings.
    
    Methods:
    - load_and_parse_configs_file(): Load in a config file and parse it into an exportable Configs object.
    - parse_command_line(): Parse command line arguments into an exportable Configs object.
    - save_to_configs_file(): Save the current config settings to a config file.
    """

    def __init__(self):
        """
        Class Constructor. Initializes the ConfigParser object.

        Potential global issues:
         - Thread safety concerns across all methods
         - Memory leaks from improper resource handling
         - Inconsistent state between file and command line configs
         - Error handling and logging consistency
         - Security audit logging requirements
         - Configuration change notification system
         - Configuration version management
         - Backup and recovery procedures
         - Configuration inheritance and override rules
         - Default value handling strategy
        """
        self._ROOT_DIR = Path(__file__).parent.parent
        if not self._ROOT_DIR.exists():
            raise FileNotFoundError(f"Cannot find the root directory:{self._ROOT_DIR}")

        self.configs_file_path: Path = self._ROOT_DIR / "configs.yaml"

        if not self.configs_file_path.exists():
            make_config_file = input("""
                Cannot find configs.yaml in root directory. Would you like to create a new config file from the program defaults? (Y/n): 
                """)
            if make_config_file.lower() == "y":
                self.save_current_config_settings_to_configs_file(Configs())
            else:
                raise FileNotFoundError(f"Config file not found at {self.configs_file_path}")


    def load_and_parse_configs_file(self) -> Configs:
        """
        Load in a config file and parse it into an exportable Configs object.
        
        Args:
            None.

        Returns:
            Configs: An exportable Configs object containing the parsed settings from the config file.

        Potential Issues:
            1. File-related issues:
                - Config file not found.
                - Config file cannot be loaded.
                - Config file is not the correct type of file.
                - Config file is empty.
                - Config file is malformed/encoded.
                - More than one config file is present in the current working directory.

            2. Permission issues:
                - Lack of permissions to read the config file.

            3. Argument validation issues:
                - No arguments in config file are valid, where validity refers to:
                    - Correct Type
                    - Correct Value Range
                    - Correct Syntax (e.g. correct URL format)
                - Some arguments in config file are not valid.
                - Pairwise arguments (e.g. API key and API URL) in config file are not valid.
                - Extra arguments are present in the config file.
                - Config values don't meet minimum/maximum constraints.

            4. Version and compatibility issues:
                - Version mismatch between saved config format and current expected format.
                - Platform-specific path separator issues (e.g. Windows vs. Unix).

            5. Data integrity issues:
                - Circular references in config file.
                - Interdependent config values have conflicting settings.
                - Default values overriding explicit NULL values.

            6. Environmental issues:
                - Environment variable substitutions in config file failing.
                - Network timeout when reading from network drives or remote locations.

            7. Resource-related issues:
                - Memory overflow when loading very large config files.
                - Race conditions if multiple processes try to read the config file simultaneously.

            8. Security-related issues:
                - Arguments containing sensitive data exposed in process list
                - Command injection vulnerabilities in argument parsing.
        """
        if not self.configs_file_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.configs_file_path}")

        try:
            with open(self.configs_file_path, 'r', encoding='utf-8') as file:
                configs_dict = yaml.safe_load(file)

        except PermissionError:
            raise PermissionError(f"Permission denied when accessing config file at {self.configs_file_path}")
        except IOError as e:
            raise PermissionError(f"Unable to read config file at {self.configs_file_path}: {e}")
        except yaml.YAMLError:
            raise yaml.YAMLError(f"Invalid YAML format in config file at {self.configs_file_path}")

        if configs_dict is None:
            raise ValueError(f"Config file at {self.configs_file_path} is empty or invalid YAML")

        try:
            return Configs(**configs_dict)
        except TypeError as e:
            raise ValueError(f"Invalid configuration structure: {e}")


    def check_if_the_program_was_started_from_command_line() -> bool:
        """Checks if the program was started from the command line."""
        return len(sys.argv) > 1


    def parse_command_line(self, args: Namespace) -> Configs:
        """
        Parse command line arguments into an exportable Configs object.
        
        Args:
            None.

        Returns:
            Configs: An exportable Configs object containing the parsed settings from the config file.

        Potential Issues:
            1. Argument Validity:
                - No arguments are provided.
                - Some arguments are not valid.
                - Missing required arguments.
                - Extra arguments are present.
                - Pairwise arguments (e.g. API key and API URL) are not valid.
                - Duplicate argument definitions with different values.

            2. Parsing and Formatting:
                - Unicode/special characters in arguments causing parsing errors.
                - Invalid escape sequences in argument values.
                - Case sensitivity issues in argument names.
                - Platform-specific argument parsing differences.

            3. Environmental and Path Issues:
                - Environment variables referenced in arguments don't exist.
                - Relative path resolution failures.

            4. Security Concerns:
                - Command injection vulnerabilities in argument parsing.
                - Arguments containing sensitive data exposed in process list.

            5. Data Integrity:
                - Circular references in command line argument values.

            6. Resource Limitations:
                - Buffer overflow in very long argument values.
        """
        return Configs(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            max_memory=args.max_memory,
            conversion_timeout=args.conversion_timeout,
            log_level=args.log_level,
            max_connections_per_api=args.max_connections_per_api,
            max_threads=args.max_threads,
            batch_size=args.batch_size,
            llm_api_key=args.llm_api_key,
            llm_api_url=args.llm_api_url,
            use_docintel=args.use_docintel,
            docintel_endpoint=args.docintel_endpoint,
            version=args.version,
            pool_refresh_rate=args.pool_refresh_rate,
            pool_health_check_rate=args.pool_health_check_rate
        )

    def save_current_config_settings_to_configs_file(self, configs: Configs) -> None:
        """
        Save the current config settings to a config file.

        Args:
            configs (Configs): The current config settings to be saved

        Returns:
            None: A config file with the same values currently being used by the program is saved to the current working directory.

        Potential Issues:
            1. File-related issues:
                - Config file to be saved to is not found.
                - Config file to be saved to is not the correct type of file.
                - More than one config file is present in the current working directory.
                - Invalid characters in file path.
                - Config file size exceeding system limits.

            2. Permission and access issues:
                - Config file to be saved to is not writable.
                - Permission escalation attempts through symlinks.
                - Inability to maintain file permissions/ownership.
                - File locking issues in multi-process scenarios.

            3. Data integrity and validation:
                - No arguments are provided.
                - Some arguments are not valid.
                - Configs to be saved have the same values as those currently in the config file.
                - Config values containing special characters not properly escaped.
                - Inability to serialize complex objects.
                - Insufficient entropy for generating secure values.

            4. File system and storage issues:
                - Disk space limitations preventing save.
                - File system becoming read-only during save.
                - Partial write leaving corrupt config file.
                - File system corruption during save.
                - File system journaling issues.
                - Network drive disconnection during save.
                - File system encryption conflicts.

            5. Operational and process issues:
                - Backup/temporary file creation failures.
                - Save operation timeout.
                - System crash during save corrupting file.
                - Version control conflicts.
                - Race conditions between validation and save operations.
                - Transaction atomicity failures during save.
        """
        try:
            with open(self.configs_file_path, "w", encoding='utf-8') as file:
                yaml.safe_dump(
                    configs.model_dump(), 
                    file, 
                    default_flow_style=False
                )
        except IOError as e:
            print(f"Error writing to config file: {e}")
        except yaml.YAMLError as e:
            print(f"Error dumping config to YAML: {e}")


def main():
    parser = argparse.ArgumentParser(
        prog="convert_to_txt_based_on_mime_type",
        formatter_class=argparse.MetavarTypeHelpFormatter,
        description="Convert files into text documents based on their MIME type. Format of the text documents will be Markdown."
    )
    parser.add_argument("input-folder", 
                        type=str, 
                        default="input", 
                        help="Path to the folder containing the files to be converted. Defaults to 'input', the name of the input folder in the working directory.")
    parser.add_argument("output-folder", 
                        type=str, 
                        default="output", 
                        help="Path to the folder where the converted files will be saved. Defaults to 'output', the name of the output folder in the working directory.")
    parser.add_argument("--max-memory", 
                        type=int, 
                        default=1024, 
                        help="Maximum amount of memory in Megabytes the program can use at any one time. Defaults to 1024 MB.")
    parser.add_argument("--conversion-timeout", 
                        type=int, 
                        default=30, 
                        help="Maximum amount of time in seconds an API-bounded conversion can run before it is terminated. Defaults to 30 seconds.")
    parser.add_argument("--log-level", 
                        type=str, 
                        default="INFO", 
                        help="Level of logging to be used. Defaults to 'INFO'.")
    parser.add_argument("--max-connections-per-api", 
                        type=int, 
                        default=3, 
                        help="Maximum number of concurrent API connections the program can have at any one time. Defaults to 3.")
    parser.add_argument("--max-threads", 
                        type=int, 
                        default=4, 
                        help="Maximum number of threads to be used for processing the program can use at any one time. Defaults to 4.")
    parser.add_argument("--batch-size", 
                        type=int, 
                        default=1024, 
                        help="Number of files to be processed in a single batch. Defaults to 1024.")
    parser.add_argument("--llm-api-key", 
                        type=str, 
                        default="abcde123456", 
                        help="API key for the LLM API. Defaults to 'abcde123456'.")
    parser.add_argument("--llm-api-url", 
                        type=str, 
                        default="www.example.com", 
                        help="URL for the LLM API. Defaults to 'www.example.com'.")
    parser.add_argument("--use-docintel", 
                        action="store_true", 
                        help="Use Document Intelligence to extract text instead of offline conversion. Requires a valid Document Intelligence Endpoint. Defaults to False.")
    parser.add_argument("--docintel-endpoint", 
                        type=str, 
                        default="www.example2.com", 
                        help="Document Intelligence Endpoint. Required if using Document Intelligence. Defaults to 'www.example2.com'.")
    parser.add_argument("-v", "--version", 
                        action="version", 
                        version="%(prog)s 1.0.0", 
                        help="Show program's version number and exit.")
    parser.add_argument("--pool-refresh-rate", 
                        type=int, 
                        default=60, 
                        help="Refresh rate in seconds for refreshing resources in the Pools. Defaults to 60 seconds.")
    parser.add_argument("--pool-health-check-rate", 
                        type=int, 
                        default=30, 
                        help="Health check rate in seconds for checking resources in the Pools. Defaults to 30 seconds.")
    args = parser.parse_args()

    config_parser = ConfigParser()
    configs = config_parser.parse_command_line(args)

if __name__ == "__main__":
    main()