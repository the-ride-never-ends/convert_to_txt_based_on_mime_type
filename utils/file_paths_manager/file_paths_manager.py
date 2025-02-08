
from enum import Enum
from pathlib import Path
from typing import Generator

from pydantic import BaseModel, Field, field_validator
from pydantic_models.configs import Configs
from pydantic_models.types.valid_path import ValidPath


class SupportedTypes(str, Enum):
    """Supported file types"""
    PDF = ".pdf"
    JPEG = ".jpeg"
    JPG = ".jpg"
    PNG = ".png"
    TIFF = ".tiff"




class FilePath(BaseModel):
    """
    
    """
    file_path: ValidPath = Field(..., description="The path to the file")

    @field_validator("file_path", mode="after")
    def check_if_file_exists(self, v: Path):
        pass


class FilePathMetadata(BaseModel):
    """
    
    """
    pass


class FilePathAndMetadata(BaseModel):
    """
    
    """
    pass


class FilePathBatch():
    """
    
    """
    _configs: Configs = Field(..., description="Configuration settings")
    batch: list[FilePathAndMetadata] = Field(..., description="List of file paths and their metadata")

    def __init__(self, **data):
        super().__init__(**data)
        

class FilePathsManager():
    """
    
    """
    def __init__(self, configs: Configs):
        self.configs = configs
        self.supported_types = [type_ for type_ in SupportedTypes]


    def get_file_paths_from_input_dir(self) -> list[FilePath]:
        """
        """
        pass


    def make_file_path_metadata(self) -> FilePathMetadata:
        """
        """
        pass


    def dispatch_to_external_resource_manager(self) -> FilePathBatch:
        """
        """
        pass
    