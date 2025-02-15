
from enum import Enum
from pathlib import Path
from typing import Callable, Coroutine, Optional



from pydantic import BaseModel, Field, PrivateAttr


from pydantic_models.resource.api_connection import ApiConnection
from converter_system.core.pipeline import Pipeline


class FunctionType(Enum):
    """Enum for function types"""
    SAVE = "save"
    LOAD = "load"
    CONVERT = "convert"


class FunctionDictionary(BaseModel):
    """
    A container for a function and resources allocated to it.
    The function can be a callable or a coroutine. The resources are allocated based on the function type.
    
    Example:
        func_dict = FunctionDictionary(**data)
        func_dict.model_dump()
        {
            "func": json.dumps,
            "kwargs": {"indent": 4},
            "thread": 1,
            "gpu_mem": 0,
            "sys_mem": 1024,
            "api_connection": 0
        }
    """
    func: Callable
    kwargs: Optional[dict] = None
    thread: Optional[int] = None
    gpu_mem: Optional[int] = None
    sys_mem: Optional[int] = None
    api_connection: Optional[int] = None



from functools import partial

class Resource(BaseModel):
    """
    A container for a series of functions and the resources needed to run them
    """
    file_path: Optional[Path] = None
    func_dict: Optional[dict[FunctionType, FunctionDictionary]] = Field(default=None, description="A dictionary of functions and their associated resources.")
    api_connection: Optional[list[ApiConnection]] = Field(default=None, description="List of API connections available for this resource.")
    total_gpu_mem: Optional[int] = Field(default=None, description="Total amount of GPU memory available for this resource.")
    total_workers: Optional[int] = Field(default=None, description="Total number of workers available for this resource.")
    total_sys_mem: Optional[int] = Field(default=None, description="Total amount of system memory available for this resource.")
    total_api_uses: Optional[int] = Field(default=None, description="Total number of API calls available for this resource.")

    data: Optional[bytes] = None
    converted_data: Optional[bytes] = None
    _pipeline: Pipeline = None

    _file_path: Optional[Path] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._pipeline = Pipeline(self)

    def release_this(self, resource_type):
        """
        Release all resources used by the indicated resource.
        """
        pass

    def request_this(self, resource_type):
        """
        Request all resources used by the indicated resource.
        """
        pass

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    # NOTE These functions are meant to be passed in a pipeline

    async def load(self) -> Optional[bytes]:
        kwargs = self.func_dict[FunctionType.LOAD].kwargs
        func = partial(self.func_dict[FunctionType.LOAD], **kwargs)
        return await func()

    async def convert(self, result) -> Optional[bytes]:
        kwargs = self.func_dict[FunctionType.CONVERT].kwargs
        func = partial(self.func_dict[FunctionType.CONVERT], **kwargs)
        try:
            return await func(result) if result else None
        except Exception as e:
            return e
        finally:
            self.total_api_uses -= self.func_dict[FunctionType.CONVERT].api_connection
            self.total_gpu_mem -= self.func_dict[FunctionType.CONVERT].gpu_mem
            self.total_sys_mem -= self.func_dict[FunctionType.CONVERT].sys_mem

    async def save(self, result) -> Optional[bytes]:
        kwargs = self.func_dict[FunctionType.SAVE].kwargs
        func = partial(self.func_dict[FunctionType.SAVE], **kwargs)
        return await func(self.converted_data) if result else None

