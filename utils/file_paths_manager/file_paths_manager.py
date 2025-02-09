
from datetime import datetime
from functools import cached_property
from enum import Enum
import hashlib
import os
from pathlib import Path
from typing import Annotated, ClassVar, Callable, Generator, Optional, Self


from utils.common.ipfs_multiformats import get_cid
from pydantic import AfterValidator, BaseModel, Field, field_validator, PrivateAttr
from pydantic_models.configs import Configs
from pydantic_models.types.valid_path import ValidPath


from logger.logger import Logger


# Calculate MD-5 checksum of a file
def md5_checksum(file_path: Path) -> str:
    hash_list = []
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class SupportedApplicationTypes(str, Enum):
    """Supported application types"""
    PDF = ".pdf"
    DOCX = ".docx"
    DOC = ".doc"
    XLS = ".xls"
    XLSX = ".xlsx"
    PPT = ".ppt"
    PPTX = ".pptx"
    ZIP = ".zip"
    RAR = ".rar"
    TXT = ".txt"
    RTF = ".rtf"
    ODT = ".odt"
    ODS = ".ods"
    ODP = ".odp"
    CSV = ".csv"
    JSON = ".json"
    XML = ".xml"
    EXE = ".exe"
    DMG = ".dmg"


class SupportedImageTypes(str, Enum):
    """Supported image types"""
    JPEG = ".jpeg"
    JPG = ".jpg"
    PNG = ".png"
    TIFF = ".tiff"
    GIF = ".gif"
    BMP = ".bmp"
    WEBP = ".webp"
    SVG = ".svg"
    ICO = ".ico"
    HEIC = ".heic"
    RAW = ".raw"


class SupportedVideoTypes(str, Enum):
    """Supported video types"""
    MP4 = ".mp4"
    AVI = ".avi"
    MOV = ".mov"
    WMV = ".wmv"
    FLV = ".flv"
    MKV = ".mkv"
    WEBM = ".webm"
    M4V = ".m4v"
    MPG = ".mpg"
    MPEG = ".mpeg"
    _3GP = ".3gp"
    TS = ".ts"
    VOB = ".vob"
    OGV = ".ogv"
    MTS = ".mts"
    M2TS = ".m2ts"


class SupportedAudioTypes(str, Enum):
    """Supported audio types"""
    MP3 = ".mp3"
    WAV = ".wav"
    AAC = ".aac"
    OGG = ".ogg"
    FLAC = ".flac"
    M4A = ".m4a"
    WMA = ".wma"
    AIFF = ".aiff"
    ALAC = ".alac"
    AMR = ".amr"
    AU = ".au"
    MID = ".mid"
    MIDI = ".midi"
    RA = ".ra"
    RM = ".rm"

class SupportedTextTypes(str, Enum):
    """Supported text types"""
    TXT = ".txt"
    RTF = ".rtf"
    HTML = ".html"
    HTM = ".htm"
    XML = ".xml"
    JSON = ".json"
    YAML = ".yaml"
    YML = ".yml"
    CSV = ".csv"
    TSV = ".tsv"
    MD = ".md"
    MARKDOWN = ".markdown"
    INI = ".ini"
    CFG = ".cfg"
    LOG = ".log"
    SQL = ".sql"
    PY = ".py"
    JS = ".js"
    CSS = ".css"
    SH = ".sh"
    BAT = ".bat"


SupportedTypes = {*SupportedApplicationTypes, *SupportedImageTypes, *SupportedVideoTypes, *SupportedAudioTypes, *SupportedTextTypes}

class MimeType(str, Enum):
    """Mime types"""
    APPLICATION = "application"
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"



def validate_file_path(value: Path) -> Path:

    # Check if it's a file-type we have a converter for.
    if value.suffix not in SupportedTypes:
        raise ValueError(f"File type {value.suffix} is not supported")

    # Check if the file exists.
    if not value.exists():
        raise ValueError(f"File {value} does not exist")

    # Check if we have read permissions for the file.
    if not os.access(value, os.R_OK):
        raise ValueError(f"Program lacks read permissions for File {value}")

    return value


class FilePath(BaseModel):
    """
    A model representing a file path.

    Attributes:
        file_path (ValidPath): The path to the file. It must meet the following criteria:
            - The path must be a valid file path.
            - The file must exist in the input directory.
            - The file must be readable.
            - The file must be of a type we have a converter for.
            - The file's size must be under the memory limit allocated to the program.
    """
    file_path: Annotated[ValidPath, AfterValidator(validate_file_path)]


class FilePathAndMetadata(BaseModel):
    """
    A model representing a file path and its associated metadata.

    Attributes:
       file_path (FilePath): The path to the file.
       cid (Optional[str]): The content identifier for the file.
    
    Properties:
       file_name (str): The name of the file.
       file_extension (str): The extension of the file.
       mime_type (MimeType): The mime type of the file, as determined by its extension.

    """
    configs: ClassVar[Configs]
    file_path: FilePath
    cid: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.cid = get_cid(self.file_path)

    @field_validator("file_path", mode="after")
    @classmethod
    def check_if_file_is_under_the_size_limit(cls, value: FilePath) -> Self:
        file_size = value.stat().st_size
        max_file_size = cls.configs.max_memory
        if file_size > max_file_size:
            raise ValueError(f"File size ({file_size} bytes) exceeds {max_file_size} bytes of memory allocated to the program.")
        return value

    @property
    def file_name(self):
        return self.file_path.stem

    @property
    def file_extension(self) -> str:
        return self.file_path.suffix

    @cached_property
    def mime_type(self) -> MimeType:
        file_extension = self.file_path.suffix.lower()
        if file_extension in SupportedApplicationTypes:
            return MimeType.APPLICATION
        elif file_extension in SupportedAudioTypes:
            return MimeType.AUDIO
        elif file_extension in SupportedImageTypes:
            return MimeType.IMAGE
        elif file_extension in SupportedTextTypes:
            return MimeType.TEXT
        elif file_extension in SupportedVideoTypes:
            return MimeType.VIDEO
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    @cached_property
    def file_size(self) -> int:
        self.file_path: Path
        return self.file_path.stat().st_size

    @cached_property
    def checksum(self) -> str:
        return md5_checksum(self.file_path)
    
    @cached_property
    def created_timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.file_path.stat().st_birthtime)

    @cached_property
    def modified_timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.file_path.stat().st_mtime)


class FilePathBatch(BaseModel):
    """
    A model representing a batch of file paths and their metadata.
    """
    configs: ClassVar[Configs] = Field(..., description="Configuration settings")
    batch: list[FilePathAndMetadata] = Field(..., description="List of file paths and their metadata")

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch:
            return self.batch.pop(0)
        raise StopIteration


class FilePathsManager():
    """
    
    """
    def __init__(self, configs: Configs):
        self.configs = configs
        self.configs.logger = Logger(self.__name__)
        self.output_cids = set()


    def get_cids_from_output_dir(self) -> set[str]:
        """
        Get CIDs from the output directory.
        This method will recursively search for files in the output directory,
            extract the CIDs from the file names, and return them as a set.

        Returns:
            set[str]: A set of CIDs from the output directory.
        """

    def get_file_paths_from_input_dir(self) -> set[FilePath]:
        """
        Get file paths from the input directory. 
        This method will recursively search for files in the input directory and return a list of file paths.

        Performs the following checks:
            Checks the file types and skips any unsupported file types.
            Checks the file size and skips any files the size allocated to the program.
            Checks the file's CID against the database and/or output folder and skips any files 
                that have already been processed.

        Returns:
           list[FilePath]: A list of file path base models created from the files in the input directory.

        Raise:
           FileNotFoundError: If the input directory does not exist or is not a directory.
           ValueError: If the input directory is empty.
        """
        pass

    def make_file_path_metadata(self, file_path: FilePath) -> FilePathAndMetadata:
        """
        Make metadata for a list of file paths.

        Args:
            file_path (FilePath): The file path to make metadata for.

        Returns:
            FilePathAndMetadata: The metadata for the file path.
        """
        pass

    def create_batch(self, batch_size: int = 1024) -> FilePathBatch:
        """
        Create a batch of file paths and their metadata.
        This method will create a batch of file paths and their metadata from the input directory.
        The batch's size is specified in the configuration settings.

        Args:
            batch_size (int): The size of the batch to create. Defaults to 1024 (paths).

        Returns:
            FilePathBatch: A batch of file paths and their metadata.
        """
        pass

    def __iter__(self):
        return self

    def __next__(self) -> FilePathBatch:
        pass

