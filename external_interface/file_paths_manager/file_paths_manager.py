from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from enum import Enum
from functools import cached_property
import hashlib
import logging
import os
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Callable, ClassVar, Generator, Iterator, Optional, Self


# External libraries
from pydantic import AfterValidator, BaseModel, computed_field, Field, field_validator, ValidationError


# Local imports
from pydantic_models.configs import Configs
from pydantic_models.types.valid_path import ValidPath
from pydantic_models.default_paths import DefaultPaths
from utils.common.get_cid import get_cid
from .file_path_and_metadata import FilePathAndMetadata






from .file_path import FilePath
from .supported_mime_types import (
    SupportedMimeTypes,
    SupportedApplicationTypes,
    SupportedAudioTypes,
    SupportedImageTypes,
    SupportedTextTypes,
    SupportedVideoTypes
)






class ComparisonFunction(ABC):
    """
    Abstract base class for file comparison functions in the pipeline.
    """

    def __init__(self, configs: Configs):
        self.configs = configs

    @abstractmethod
    def compare(self, input_path: FilePathAndMetadata, output_path: FilePathAndMetadata) -> bool:
        """Compare input and output paths."""
        pass


class ExistsComparator(ComparisonFunction):

    def __init__(self, configs: Configs):
        super().__init__(configs)

    def compare(self, input_path: FilePathAndMetadata, output_path: FilePathAndMetadata) -> bool:
        pass


class TimestampComparator(ComparisonFunction):

    def __init__(self, configs: Configs):
        super().__init__(configs)

    def compare(self, input_path: FilePathAndMetadata, output_path: FilePathAndMetadata) -> bool:
        pass



class FilePathsManager:
    """Manages file paths and their metadata for batch processing operations.

    Purpose:
        Handles discovery, validation, and metadata extraction for files in a given
        input directory, comparing them against an output directory and providing
        filtered results based on configurable comparison rules.

    Implementation Strategy:
        - Uses a pipeline of comparison functions for flexible filtering
        - Implements RAII pattern for resource management
        - Provides cross-platform path handling via pathlib
        - Supports concurrent processing with proper resource cleanup
        - Uses context manager protocol for safe resource handling

    Inputs:
        - Input directory path
        - Output directory path
        - List of comparison functions
        - Configuration options for filtering and logging

    Outputs:
        - Iterator of FileAndMetadata objects
        - Logging of invalid/inaccessible paths

    Attributes:
        input_dir (Path): Root directory for input files
        output_dir (Path): Directory containing processed files
        comparison_pipeline (List[ComparisonFunction]): Ordered list of comparison functions
        logger (logging.Logger): Logger instance
        _executor (ThreadPoolExecutor): Thread pool for concurrent operations
    
    Methods:
        __init__: Initialize the manager with paths and configuration
        __aenter__: Async Context manager entry
        __aexit__: Async Context manager exit and cleanup
        get_inputs: Main method to traverse and process files
        extract_metadata: Extract metadata for a given file
        requires_processing: Check if a file needs processing
        cleanup: Perform resource cleanup


    Example:
        >>> async with FilePathsManager(configs) as fpm:
        ...     async for file in fpm.get_inputs():
        ...         needs_processing = fpm.requires_processing(file)
        ...         if needs_processing:
        ...             fpm.extract_metadata_queue.put(file)
        ... 
        ...      async for file_path_and_metadata in extract_metadata():
        ...         fpm.output_queue.put(file_path_and_metadata)
        ... 
        ...      async for batch in fpm.output_queue:
                    fpm.send_to_external_file_manager(batch)
    """

    def __init__(self, configs: Configs) -> None:
        """Initialize the FilePathsManager.
        
        Args:
            configs: A configuration object
            input_dir: Root directory to search for files
            output_dir: Directory containing processed files
            comparison_functions: List of functions for file comparison pipeline
            log_level: Logging level for operations
            max_workers: Maximum number of concurrent workers

        Example:
            >>> fpm = FilePathsManager(configs)
        """
        self.configs = configs

        self._input_folder: Path = Path(configs.input_folder) or DefaultPaths.INPUT_DIR
        self._output_folder: Path = Path(configs.output_folder) or DefaultPaths.OUTPUT_DIR
        self._comparison_pipeline: list[ComparisonFunction] = [ExistsComparator(configs), TimestampComparator(configs)]
        self._max_workers = configs.max_workers or 2
        self._max_queue_size = configs.max_queue_size or 1024
        self._max_program_memory = configs.max_program_memory

        self._process_pool_executor = ProcessPoolExecutor(max_workers=self._max_workers)
        self.logger = configs.make_logger(self.__class__.__name__)
        self.duck_db = configs.make_duck_db('file_path_manager.db')

        self.get_inputs_queue = asyncio.Queue(maxsize=self._max_queue_size)
        self.extract_metadata_queue = asyncio.Queue(maxsize=self._max_queue_size)
        self.processing_queue = asyncio.Queue(maxsize=self._max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=self._max_queue_size)


    async def __aenter__(self) -> 'FilePathsManager':
        """Enter context manager, initializing resources.

        Returns:
            Self for context manager protocol

        Example:
            >>> async with FilePathsManager(configs) as fpm:
            ...     for file in fpm.get_inputs():
            ...         process_file(file)
        """
        return await self


    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, ensuring cleanup.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred

        Example:
            >>> # Resources automatically cleaned up after context exit
            >>> async with FilePathsManager(configs) as fpm:
            ...     fpm.get_inputs()
        """
        return await self.cleanup()


    async def get_inputs(self) -> None:
        """Validate paths coming in from the input directory
            by encasing them in the Pydantic class 'FilePath'
                If they're valid, put them in the get_inputs_queue.
                If they're not log it and skip them.

        Returns:
            None: The result is that the file_path is put in the extract_metadata_queue

        Example:
            >>> async with FilePathsManager(configs) as fpm:
            ...     await fpm.get_inputs()
            ...     while not fpm.extract_metadata_queue.empty():
            ...         file_path = await fpm.extract_metadata_queue.get()
            ...         print(f"Valid file path: {file_path}")
            ...         fpm.extract_metadata_queue.task_done()
        """
        for file_path in self.scan_for_files():
            try:
                validated_path = FilePath(file_path=file_path)
                await self.extract_metadata_queue.put(validated_path)
            except ValidationError as e:
                self.logger.error(f"Invalid file path: {e}")


    def scan_for_files(self) -> Generator[Path, None, None]:
        """
        Scan the input directory for files.
        This includes sub-directories and hidden files, but not objects like symlinks and shortcuts.

        Yields:
            Path: A Path object for a file in the input directory.
        """
        try:
            for root, _, files in os.walk(self._input_folder):
                for file in files:
                    path = Path(root) / file
                    if path.is_file() and not path.is_symlink():
                        yield path
        except Exception as e:
            self.logger.error(f"Error scanning input directory: {e}")
            raise e


    async def extract_metadata(self) -> AsyncGenerator[FilePathAndMetadata, None]:
        """Extract both basic and content-based metadata from a file.
        
        Yields:
            A FilePathAndMetadata object that contains a path and content metadata

        Example Output:
            >>> async with FilePathsManager(configs) as fpm:
            ...     async for path in fpm.get_inputs():
            ...         fpm.metadata_generation_queue.put(path)
            ...         async for path_with_metadata in fpm.extract_metadata():
            ...             fpm.output_queue.add(path_with_metadata)
        """
        await self.get_inputs()
        while not self.extract_metadata_queue.empty():
            file_path: FilePath = await self.extract_metadata_queue.get()
            try:
                file_path_and_metadata = FilePathAndMetadata(file_path=file_path, max_program_memory=self._max_program_memory)
                await self.processing_queue.put(file_path_and_metadata)
                self.extract_metadata_queue.task_done()
            except ValidationError as e:
                self.logger.error(f"Could not create FilePathAndMetadata object for '{file_path.file_path}': {e}")


    async def requires_processing(self, input_path: FilePathAndMetadata) -> bool:
        """Determine if a file requires processing based on comparison pipeline.
        
        Args:
            input_path: Path to input file

        Returns:
            True if file needs processing, False otherwise.
            If True, the method also outputs the input file path object.
        
        Example:
            >>> async with FilePathsManager(configs) as fpm:
            ...     needs_processing = fpm.requires_processing(input_path)
            ...     if needs_processing:
            ...         manager.get_inputs(input_path)
        """

    def __rshift__(self, item):
        return self.bind(item)


    async def send_to_external_resource_manager(self) -> AsyncGenerator[list[FilePathAndMetadata], None]:
        """
        Package the file paths currently in the output_queue and send them to the External Resource Manager.

        Example:
            >>> async with FilePathsManager(configs) as fpm:
            ...     needs_processing = fpm.requires_processing(
            ...         input_path,
            ...         output_path
            ...     )
            ...    
        """

    async def cleanup(self) -> None:
        """Clean up resources, ensuring proper release of system resources.

        This should:
            Save all the queues to the 
        
        Example:
            >>> fpm = FilePathsManager(input_dir, output_dir, comparators)
            >>> try:
            ...     fpm.get_inputs()
            ... finally:
            ...     fpm.cleanup()
        """
        pass

configs = Configs()
fpm = FilePathsManager(configs)

