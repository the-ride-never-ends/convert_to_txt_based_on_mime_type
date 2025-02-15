
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import tempfile


import pytest


from pydantic_models.configs import Configs
from external_interface.file_paths_manager.file_paths_manager import (
    FilePathsManager, ExistsComparator, TimestampComparator
)

from logger.logger import Logger
logger = Logger(__name__)


# 1. Test Instantiation
@pytest.mark.parametrize("configs", [Configs(batch_size=10, input_folder="input", output_folder="output", max_workers=4, max_queue_size=2048)])
def test_file_paths_manager_initialization(configs: Configs):
    file_manager = FilePathsManager(configs)

    # Test input and output folders
    assert file_manager._input_folder == Path("input")
    assert file_manager._output_folder == Path("output")

    # Test comparison pipeline
    assert len(file_manager._comparison_pipeline) == 2
    assert isinstance(file_manager._comparison_pipeline[0], ExistsComparator)
    assert isinstance(file_manager._comparison_pipeline[1], TimestampComparator)

    # Test max workers and max queue size
    assert file_manager._max_workers == 4
    assert file_manager._max_queue_size == 2048

    # Test process pool executor
    assert isinstance(file_manager._process_pool_executor, ProcessPoolExecutor)
    assert file_manager._process_pool_executor._max_workers == 4

    # Test logger
    assert file_manager.logger is not None
    assert callable(file_manager.logger.info)
    assert callable(file_manager.logger.error)

    # Test DuckDB connection
    assert file_manager.duck_db is not None
    assert hasattr(file_manager.duck_db, 'execute')
    assert hasattr(file_manager.duck_db, 'fetchall')

    # Test asyncio queues
    assert isinstance(file_manager.get_inputs_queue, asyncio.Queue)
    assert isinstance(file_manager.extract_metadata_queue, asyncio.Queue)
    assert isinstance(file_manager.output_queue, asyncio.Queue)
    assert file_manager.get_inputs_queue.maxsize == 2048
    assert file_manager.extract_metadata_queue.maxsize == 2048
    assert file_manager.output_queue.maxsize == 2048



# 2. Test File Detection
# Criteria: All files in the input folder and sub-folders must be detected.
# NOTE We assume that the input folder contains files and sub-folders.



@pytest.fixture
def test_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        os.makedirs(os.path.join(tmpdir, "subfolder1"))
        os.makedirs(os.path.join(tmpdir, "subfolder2", "subsubfolder"))

        # Create some files
        open(os.path.join(tmpdir, "file1.txt"), "w").close()
        open(os.path.join(tmpdir, "subfolder1", "file2.txt"), "w").close()
        open(os.path.join(tmpdir, "subfolder2", "file3.txt"), "w").close()
        open(os.path.join(tmpdir, "subfolder2", "subsubfolder", "file4.txt"), "w").close()

        yield Path(tmpdir)

import os
import pytest
from pathlib import Path

@pytest.fixture
def test_files(tmp_path: Path):
    # Create temporary test files
    valid_path1 = tmp_path / "valid1.txt"
    valid_path2 = tmp_path / "valid2.txt"
    invalid_path = tmp_path / "invalid.IDENTIFIER"  # unsupported extension

    # Create the actual files
    valid_path1.write_text("test content")
    valid_path2.write_text("test content")
    invalid_path.write_text("test content")

    # Set read permissions explicitly
    os.chmod(valid_path1, 0o644)  # User read/write, group/others read
    os.chmod(valid_path2, 0o644)
    os.chmod(invalid_path, 0o000)  # No permissions (should fail validation)

    # Return a list of Paths
    return [valid_path1, valid_path2, invalid_path]


def test_file_detection(test_directory: Path):
    configs = Configs(batch_size=10, input_folder=test_directory, output_folder="output", max_workers=4, max_queue_size=2048)
    file_manager = FilePathsManager(configs)

    # Call the method that detects and gets files (assuming it's called scan_for_files)
    # Now it returns a generator instead of a list
    file_generator = file_manager.scan_for_files()

    test_directory = Path(test_directory)
    # Define the expected files
    expected_files = {
        test_directory / "file1.txt",
        test_directory / "subfolder1" / "file2.txt",
        test_directory / "subfolder2" / "file3.txt",
        test_directory / "subfolder2" / "subsubfolder" / "file4.txt"
    }

    # Use a set to collect detected files
    detected_files = set()

    # Iterate through the generator
    for file in file_generator:
        detected_files.add(file)

    # Check if all expected files are detected
    assert detected_files == expected_files

    # Optionally, you can check the count of files
    assert len(detected_files) == len(expected_files)


def test_empty_directory(test_directory: Path):
    # Remove all files and subdirectories
    for root, dirs, files in os.walk(test_directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

    configs = Configs(batch_size=10, input_folder=test_directory, output_folder="output", max_workers=4, max_queue_size=2048)
    file_manager = FilePathsManager(configs)

    # Assuming scan_for_files now returns a generator
    file_generator = file_manager.scan_for_files()

    # Convert generator to list to check its length
    detected_files = list(file_generator)

    assert len(detected_files) == 0

    # Alternatively, if you want to avoid converting to a list:
    # assert sum(1 for _ in file_generator) == 0

def test_hidden_files(test_directory: Path):
    # Create a hidden file
    hidden_file = test_directory / ".hidden_file.txt"
    open(hidden_file, "w").close()

    configs = Configs(batch_size=10, input_folder=test_directory, output_folder="output", max_workers=4, max_queue_size=2048)
    file_manager = FilePathsManager(configs)

    # Assuming scan_for_files now returns a generator
    file_generator = file_manager.scan_for_files()

    # Collect all files from the generator
    detected_files = [file for file in file_generator]

    assert hidden_file in detected_files

    # Optional: You might want to add an assertion to check if other expected files are also present
    # For example:
    assert len(detected_files) > 1  # Ensures other files are also detected

# NOTE Had to turn on developer mode in windows to test this.
def test_ignore_symlinks(test_directory: Path):
    # Create a symlink
    os.symlink(test_directory / "file1.txt", test_directory / "symlink.txt")

    configs = Configs(batch_size=10, input_folder=test_directory, output_folder="output", max_workers=4, max_queue_size=2048)
    file_manager = FilePathsManager(configs)

    # Assuming scan_for_files now returns a generator
    file_generator = file_manager.scan_for_files()

    # Collect all files from the generator
    detected_files = [file for file in file_generator]

    # Change the assertion to verify that the symlink is NOT in the detected files
    assert test_directory / "symlink.txt" not in detected_files

    # Optional: Add an additional assertion to verify the target file IS detected
    assert test_directory / "file1.txt" in detected_files

import pytest
import asyncio
from pathlib import Path
from pydantic import ValidationError

from external_interface.file_paths_manager.file_path import FilePath
from external_interface.file_paths_manager.file_paths_manager import FilePathsManager


@pytest.mark.asyncio
async def test_get_inputs(monkeypatch: pytest.MonkeyPatch, test_files):

    # Mock FilePathsManager and its methods
    class MockFilePathsManager(FilePathsManager):
        def __init__(self):
            self.extract_metadata_queue = asyncio.Queue()
            self.logger = MockLogger()

        def scan_for_files(self):
            yield from test_files

    class MockLogger:
        def __init__(self):
            self.error_messages = []

        def error(self, message):
            self.error_messages.append(message)

    # Create instance of mock FilePathsManager
    fpm = MockFilePathsManager()

    # Mock FilePath validation
    def mock_filepath_validation(file_path):
        if "invalid" in str(file_path):
            errors = [{'loc': ('file_path',), 'msg': r'^Invalid file path.*', 'type': 'value_error'}]
            raise ValidationError(errors, FilePath)
        return FilePath(file_path=file_path)

    # Patch FilePath with mock validation
    monkeypatch.setattr("external_interface.file_paths_manager.file_path.FilePath", mock_filepath_validation)

    # Run the get_inputs method
    await fpm.get_inputs()

    # Check if valid paths were added to extract_metadata_queue
    assert fpm.extract_metadata_queue.qsize() == 2

    # Check error was logged
    assert len(fpm.logger.error_messages) == 1
    assert "Invalid file path" in fpm.logger.error_messages[0]

    # Check if the paths in extract_metadata_queue are correct
    paths = []
    while not fpm.extract_metadata_queue.empty():
        paths.append(await fpm.extract_metadata_queue.get())

    valid_test_files = [f for f in test_files if "invalid" not in str(f)]
    assert all(FilePath(file_path=f) in paths for f in valid_test_files)

    # Check if invalid path was logged
    assert len(fpm.logger.error_messages) == 1
    assert "Invalid file path" in fpm.logger.error_messages[0]

    # Check if get_inputs_queue is empty after processing
    assert fpm.extract_metadata_queue.empty()

@pytest.mark.asyncio
async def test_get_inputs_empty_directory():

    class MockLogger:
        def __init__(self):
            self.error_messages = []

        def error(self, message):
            self.error_messages.append(message)

    class MockEmptyFilePathsManager(FilePathsManager):
        def __init__(self):
            self.get_inputs_queue = asyncio.Queue()
            self.extract_metadata_queue = asyncio.Queue()
            self.logger = MockLogger()

        def scan_for_files(self):
            return []

    fpm = MockEmptyFilePathsManager()
    await fpm.get_inputs()

    assert fpm.extract_metadata_queue.empty()
    assert fpm.get_inputs_queue.empty()

@pytest.mark.asyncio
async def test_get_inputs_all_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):

    class MockLogger:
        def __init__(self):
            self.error_messages = []

        def error(self, message):
            self.error_messages.append(message)

    class MockInvalidFilePathsManager(FilePathsManager):
        def __init__(self):
            self.get_inputs_queue = asyncio.Queue()
            self.extract_metadata_queue = asyncio.Queue()
            self.logger = MockLogger()

        def scan_for_files(self):
            return [Path("/invalid/path1.txt"), Path("/invalid/path2.txt")]

    fpm = MockInvalidFilePathsManager()

    def mock_invalid_filepath_validation(path):
        raise ValidationError("Invalid path")

    monkeypatch.setattr("external_interface.file_paths_manager.file_path.FilePath", mock_invalid_filepath_validation)

    await fpm.get_inputs()

    assert fpm.extract_metadata_queue.empty()
    assert len(fpm.logger.error_messages) == 2
    assert all("Invalid file path" in msg for msg in fpm.logger.error_messages)

from external_interface.file_paths_manager.file_path_and_metadata import FilePathAndMetadata


@pytest.mark.asyncio
async def test_extract_metadata(test_directory: Path):
    configs = Configs(batch_size=10, input_folder=test_directory, output_folder="output", max_workers=4, max_queue_size=2048)

    fpm = FilePathsManager(configs)

    await fpm.extract_metadata()

    assert fpm.extract_metadata_queue.empty()
    assert fpm.processing_queue.qsize() == 4
    check_set = set()

    while not fpm.processing_queue.empty():
        path_with_metadata: FilePathAndMetadata = await fpm.processing_queue.get()

        assert isinstance(path_with_metadata, FilePathAndMetadata)

        assert isinstance(path_with_metadata.cid, str)
        assert len(path_with_metadata.cid) == 59

        assert path_with_metadata.file_name
        assert path_with_metadata.file_extension
        assert path_with_metadata.mime_type
        assert path_with_metadata.file_size == 0
        assert path_with_metadata.checksum
        assert path_with_metadata.created_timestamp
        assert path_with_metadata.modified_timestamp

        cid = path_with_metadata.cid
        logger.debug(f"cid: {cid}")
        check_set.add(cid)
    
    # 3. Test Repeat File Detection
    # Criteria: Files with the same name in different sub-folders must be detected.
    # CID generation
    # NOTE Because files in sub-folders can have the same name, we base the CID off of the whole path.

    # Make sure all CIDs are unique
    logger.debug(f"check_set: {check_set}")
    assert len(check_set) == 4

    # 4. Test Repeat File Prevention
    # Criteria: Files with the same name in different sub-folders must be prevented from being in the same batch.


# 4. Test Batch Generation
# Criteria: batches must of size to equal or lesser than the batch size specified in the configs. 
# If the batch size is larger than the total files, all files should be in one batch.
# If the batch size is smaller than the total number files, each batch must NOT contain repeat files.

# 5. Test File Attribute Generation
# Criteria: Each file must have a CID, a path, and a batch number.

# 6. Handle Validation

# 7. Cross-Platform Support for Windows and Linux.

# 8. Invalid Paths are logged and ignored.

# 9. Inaccessible Paths are logged and ignored.

# 11. Files Larger than the memory allocated to the program.