import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
import hashlib
import logging
import os
import mimetypes
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Annotated, Any, Callable, Coroutine, Generic, IO, Iterable, Never, Optional, TypeVar
from urllib.parse import urlparse
from uuid import UUID

from multiprocessing.pool import Pool, ThreadPool
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Lock


import aiohttp
import duckdb
import requests
from markitdown import MarkItDown, FileConversionException, UnsupportedFormatException
from playwright.async_api import (
    Response as PlaywrightResponse,
    async_playwright, 
    Browser,
    BrowserContext,
    Page,
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    Playwright,
    PlaywrightContextManager,
)
from pydantic import BaseModel, Field, HttpUrl, EmailStr, PrivateAttr


from utils.llm_api_manager.llm_api_manager import ApiConnection, LlmApiManager
from pydantic_models.configs import Configs
from logger.logger import Logger
logger = Logger(__name__)




def _count_number_of_camel_case_words(html: str) -> int:
    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', html)
    
    # Regular expression for camel case words
    camel_case_pattern = r'([A-Z][a-z]+){2,}'
    
    # Find all matches
    matches = re.findall(camel_case_pattern, text)
    
    # Return the count of matches
    return len(matches)



# DocumentConverterResult: TypeAlias = Any

"""
Route the input to the appropriate converter based on its ending.

Args:
    input (str): The filename to be routed.

Returns:
    The mime-type of the file.
"""

class DocumentConverterResult(BaseModel):
    title: Optional[str] = None
    text_content: str = ""


class MachineLearningModel:
    pass







# class MarkItDownAsync(MarkItDown):
#     """
#     A modified version of MarkItDown that supports asynchronous use.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._asyncio_lock = asyncio.Lock()
#         self._concurrency_limit: int = kwargs.get("concurrency_limit", 5)
#         self._aiohttp_session = None
#         self._browser: Browser = None
#         self._playwright_dom_manipulation_ruleset: dict[str, Coroutine | list[Coroutine]] = None
#         self._we_have_async: bool = None

#         try: # Importing these should tell is whether or not we have async functionality.
#             import playwright
#             import aiohttp
#             self._we_have_async = True
#         except:
#             self._we_have_async = False

#         if self._we_have_async:

#             aiohttp_session: aiohttp.ClientSession = kwargs.get("aiohttp_session", None)
#             self._aiohttp_session = aiohttp.ClientSession() if aiohttp_session is None else aiohttp_session

#             playwright_context_manager: PlaywrightContextManager = kwargs.get("playwright_context_manager", None)

#             if playwright_context_manager is None:
#                 print("Warning: No Playwright context manager provided. Loading URLs with Playwright will be disabled.")
#             else:
#                 playwright_configs: Configs = kwargs.get("playwright_configs", None)
#                 if playwright_configs is not None:
#                     self._browser = playwright_context_manager.chromium.launch(**playwright_configs)
#                 else:
#                     self._browser = playwright_context_manager.chromium.launch(headless=True)

#     @classmethod
#     def enter(cls, *args, **kwargs):
#         instance = cls(*args, **kwargs)
#         return instance

#     async def exit(self):
#         if self._aiohttp_session is not None:
#             await self._aiohttp_session.close()
#         if self._browser is not None:
#             await self._browser.close()

#     def raise_runtime_error_if_no_async(self):
#         if not self._we_have_async:
#             raise RuntimeError("Async functionality is not available. Please install the required dependencies.")

#     async def async_convert(self, 
#                       source: str | aiohttp.ClientResponse | Path | requests.Response, 
#                       **kwargs: Any
#                     ) -> DocumentConverterResult:
#         """
#         Args:
#             source: can be a string representing one of the following:
#                 - a local path to a file, either as string or a pathlib.Path object
#                 - a url string
#                 - a requests.response object
#                 - a aiohttp.ClientResponse object
#             extension: specifies the file extension to use when interpreting the file. 
#                 If None, infer from source (path, uri, content-type, etc.)

#         Returns
#            DocumentConverterResult: A pydantic model containing the result of converting a document to text.
#         """
#         self.raise_runtime_error_if_no_async()

#         # Local path or url
#         if isinstance(source, str):
#             if (
#                 source.startswith("http://")
#                 or source.startswith("https://")
#                 or source.startswith("file://")
#             ):
#                 if self._aiohttp_session is None:
#                     return self.convert_url(source, **kwargs)
#                 else:
#                     return await self.async_convert_url(source, **kwargs)
#             else:
#                 return self.convert_local(source, **kwargs)

#         # Request response
#         elif isinstance(source, requests.Response):
#             return self.convert_response(source, **kwargs)
#         elif isinstance(source, Path):
#             return self.convert_local(source, **kwargs)
#         elif isinstance(source, aiohttp.ClientResponse):
#             return self.convert_aiohttp_response(source, **kwargs)
#         elif isinstance(source, PlaywrightResponse) and self._browser is not None:
#             return self.convert_playwright_response(source, **kwargs)
#         else:
#             raise ValueError(f"Unsupported source type: {type(source)}")


#     async def async_convert_url(
#         self, url: str, **kwargs: Any
#     ) -> DocumentConverterResult:
#         self.raise_runtime_error_if_no_async()

#         # Send a HTTP request to the URL
#         with self._aiohttp_session.get(url, stream=True) as response:
#             response: aiohttp.ClientResponse
#             response.raise_for_status()
#             text_response = await response.text()
#         try:
#             # Check if we need to render the page with Playwright.
#             if self._figure_out_whether_we_need_to_render_this_url_with_playwright(text_response) and self._browser is not None:
#                 return await self._convert_url_with_playwright(url, **kwargs)
#             else:
#                 return self.convert_aiohttp_response(response, **kwargs)
#         except: # If we can't parse the HTML, then it's probably not a webpage.
#             return self.convert_aiohttp_response(response, **kwargs)


#     async def _convert_url_with_playwright(self, url: str, **kwargs: Any) -> DocumentConverterResult:
#         # NOTE Since we always call aiohttp before right this, we can assume that we'll never get a 404 here.
#         page: Page = await self._browser.new_page()
#         _page = await page.goto(url)
#         # if self._playwright_dom_manipulation_ruleset:
#         #     if url in self._playwright_dom_manipulation_ruleset:
#         #         for coroutine in self._playwright_dom_manipulation_ruleset[url]:
#         #             await coroutine(page)
#         if _page is not None:
#             await page.wait_for_load_state('networkidle')
#             response = await _page.request.response()
#             try:
#                 return await self.convert_playwright_response(response, page, **kwargs)
#             finally:
#                 await page.close()
#         else:
#             return None

#     # TODO Make this more robust.
#     def _figure_out_whether_we_need_to_render_this_url_with_playwright(self, html: str) -> bool:
#         # Count how many div HTML tags there are.
#         num_of_divs = html.lower().count('<div')

#         # Check if there's a lot of camel case words in the webpage.
#         num_of_camel_case_words = _count_number_of_camel_case_words(html)

#         return True if num_of_camel_case_words >= num_of_divs else False


#     def _read_the_extension_from_the_path(self, extensions: list[str], url: str) -> None:
#         base, ext = os.path.splitext(urlparse(url).path)
#         self._append_ext(extensions, ext)


#     async def _guess_from_the_mimetype(self, 
#                         extensions: list[str], 
#                         response: aiohttp.ClientResponse | PlaywrightResponse
#                         ) -> None:
#         if isinstance(response, PlaywrightResponse):
#             content_type = response.content_type.split(";")[0] if response.content_type else ""
#         else:
#             content_type = response.headers.get("content-type", "").split(";")[0]
#         self._append_ext(extensions, mimetypes.guess_extension(content_type))


#     async def _read_the_content_disposition_if_there_is_one(
#                         self,
#                         extensions: list[str], 
#                         response: aiohttp.ClientResponse | PlaywrightResponse, 
#                         ) -> None:
#         # Read the content disposition if there is one
#         # Get the headers from the page
#         if isinstance(response, PlaywrightResponse):
#             content_disposition = response.content_disposition if response.content_disposition else ""
#         else:
#             content_disposition = response.headers.get("content-disposition", "")
#         m = re.search(r"filename=([^;]+)", content_disposition)
#         if m:
#             base, ext = os.path.splitext(m.group(1).strip("\"'"))
#             self._append_ext(extensions, ext)


#     async def _download_and_convert_the_file(
#                         self, 
#                         fh: IO[bytes],
#                         extensions: list[str], 
#                         temp_path: str,
#                         response: aiohttp.ClientResponse | PlaywrightResponse,
#                         **kwargs: Any
#                         ) -> DocumentConverterResult:
#         chunk_size = 512  # Define chunk size (1/2 KB per write)
#         if isinstance(response, PlaywrightResponse):
#             # Get the response body
#             bytes_ = await response.body()

#             # Download the file
#             for chunk in range(0, len(bytes_), chunk_size):
#                 fh.write(bytes_[chunk:chunk+chunk_size])
#         else:
#             # Download the file
#             async for chunk in response.content.iter_chunked(chunk_size):
#                 fh.write(chunk)
#             await response.release()
#         fh.close()

#         # Use puremagic to check for more extension options
#         for g in self._guess_ext_magic(temp_path):
#             self._append_ext(extensions, g)

#         return self._convert(temp_path, extensions, url=response.url, **kwargs)

#     def prepare_a_list_of_extensions_to_try_in_order_of_priority(self,  kwargs: Any) -> list[str]:
#         # Prepare a list of extensions to try (in order of priority)
#         ext = kwargs.get("file_extension")
#         return [ext] if ext is not None else []

#     async def convert_playwright_response(self,
#         response: PlaywrightResponse, **kwargs: Any
#     ) -> DocumentConverterResult:
#         self.raise_runtime_error_if_no_async()

#         extensions = self.prepare_a_list_of_extensions_to_try_in_order_of_priority(kwargs)

#         await self._guess_from_the_mimetype(extensions, response)
#         await self._read_the_content_disposition_if_there_is_one(extensions, response)
#         self._read_the_extension_from_the_path(extensions, response.url)

#         # Save the file locally to a temporary file. It will be deleted before this method exits
#         handle, temp_path = tempfile.mkstemp()
#         fh = os.fdopen(handle, "wb")
#         result = None
#         try:
#             self._download_and_convert_the_file(                        
#                 fh, extensions, temp_path, response, **kwargs
#             )
#         finally:
#             self._clean_up(fh, response, temp_path, response)
#         return result
    
#     def _clean_up(self, fh: IO[bytes], temp_path: str, response: aiohttp.ClientResponse | PlaywrightResponse) -> None:
#         try:
#             fh.close()
#         except Exception:
#             pass
#         try:
#             response.close()
#         except:
#             pass
#         os.unlink(temp_path)

#     async def convert_aiohttp_response(
#         self, response: aiohttp.ClientResponse, **kwargs: Any
#     ) -> DocumentConverterResult:  # TODO fix kwargs type
#         self.raise_runtime_error_if_no_async()

#         extensions = self.prepare_a_list_of_extensions_to_try_in_order_of_priority(kwargs)

#         await self._guess_from_the_mimetype(extensions, response)
#         await self._read_the_content_disposition_if_there_is_one(extensions, response)
#         self._read_the_extension_from_the_path(extensions, str(response.url))

#         # Save the file locally to a temporary file. It will be deleted before this method exits
#         handle, temp_path = tempfile.mkstemp()
#         fh = os.fdopen(handle, "wb")
#         result = None
#         try:
#             self._download_and_convert_the_file(                        
#                 fh, extensions, temp_path, response, **kwargs
#             )
#         # Clean up
#         finally:
#             self._clean_up(fh, response, temp_path, response)
#         return result



def make_sha256_hash(data: Any) -> str:
    return hashlib.sha256(str(data).encode())

def make_hash_tree(*args: Iterable[Any]):
    hashed_objects = [
        make_sha256_hash(arg) for arg in args
    ]

from duckdb.typing import VARCHAR

from utils.config_parser.config_parser import ConfigParser
from utils.main.run_with_argparse import run_with_argparse

def _program_was_started_from_command_line() -> bool:
    """Checks if the program was started from the command line."""
    return len(sys.argv) > 1

from utils.common.next_step import next_step

from utils.file_paths_manager.file_paths_manager import FilePathsManager
from typing import AsyncGenerator
T = TypeVar('T')  # Generic type for pooled resources
from enum import Enum, auto

CustomClass = TypeVar("CustomClass")







class ResourceState(Enum):
    AVAILABLE = auto()
    IN_USE = auto()
    DISPOSED = auto()

class ResourceType(Enum):
    PERSISTENT = auto()  # Resources that should be kept alive (like LLMs)
    TRANSIENT = auto()   # Resources that can be destroyed and recreated (threads)
    CONSUMABLE = auto() # Resources that can be exactly once ()

import psutil

import signal 
import resource 


import subprocess as sp
import os

import functools

@functools.lru_cache(maxsize=3)
def get_gpu_memory() -> list[int]: # Available memory for each GPU (in MegaBytes)
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class SystemResource(BaseModel):
    pass


class SystemResourcesManager():

    def __init__(self, configs: Configs):
        self.configs = configs
        self.max_cores: int = psutil.cpu_count(logical=False)  # Get the number of physical CPU cores
        self.cores_in_use: int = psutil.cpu_count(logical=True)  # Get the number of logical CPU cores (including hyperthreading)
        self.memory_available: int = psutil.virtual_memory().available  # Get the amount of available memory in bytes
        self.memory_in_use: int = psutil.Process().memory_info().rss  # Get the Resident Set Size (RSS) memory used by this process
        self.gpu_memory_in_use: int = get_gpu_memory()  # Get the amount of Available memory for each GPU (in MegaBytes)

    def get_available_cores(self) -> int:
        """
        Get the number of available CPU cores from the system.

        Returns:
            int: The number of cores available as an int
            If the amount of available CPU cores is greater than the amount specified in the config file,
                then the amount specified in the config file is returned.
        """
        pass

    def get_available_memory(self) -> int:
        """
        Get the amount of available system memory from the system.

        Returns:
            SystemResource: The amount of system available in bytes as an int.
            If the amount of available system memory is greater than the amount specified in the config file,
                then the amount specified in the config file is returned.
        """
        pass

    def get_available_gpu_memory(self) -> int:
        """
        Get the amount of available GPU memory from the system.

        Returns:
            SystemResource: The amount of GPU memory available in bytes as an int.
            If the amount of available GPU memory is greater than the amount specified in the config file,
                then the amount specified in the config file is returned.
        """
        pass

    def update(self, call_every: int = 5) -> Generator[SystemResource, None, None]:
        """
        Update the system resource information.
        This method sends the current system resource information to the External Resource Manager.
        Operates based on a push system, where system resource information is automatically sent 
            to the External Resource Manager every X seconds.

        Args:
            call_every (int, optional): The interval in seconds to call the update method. Defaults to 5.
            This is used to control the frequency of updates to the External Resource Manager.

        Yields:
            SystemResource: The updated system resource information.

        """
        pass


class PooledResource(BaseModel):

    created_at: float = Field(default_factory=lambda: time.time())
    last_used_at: float = Field(default_factory=lambda: time.time())
    state: ResourceState = Field(default_factory=ResourceState.AVAILABLE)
    use_count: int = Field(default=0)
    resource_type: Optional[ResourceType] = None

    _resource: Optional[T] = PrivateAttr(default=None)

    @cached_property
    def resource(self) -> T:
        self.last_used_at = time.time()
        self.state = ResourceState.IN_USE
        return self._resource

    @resource.setter()
    def resource(self, value) -> None:
        self._resource = self.validate(value)

    @resource.deleter()
    def resource(self) -> None:
        if self.resource_type is ResourceType.PERSISTENT:
            self.reset(self)
        elif self.resource_type is ResourceType.TRANSIENT:
            self.destroy(self)
        else:
            raise AttributeError("Resource lacks a type")
        self.state = ResourceState.DISPOSED

    def create(self) -> T:
        """
        """
        pass

    def reset(self) -> bool:
        """
        Reset the resource to a clean state without destroying it.

        Returns:
            bool: True if reset was successful, False if resource needs to be re-created.
        """
        pass

    def validate(self, value) -> bool:
        """
        Validate if a given resource is healthy.
        Called when a resource the following happens:
            - The resource is returned to the Core Resource Manager.
            - The resource is first created by the External Resource Manager.
            - The resources has spent X amount of time in its respective pool.
            - The resources enters the Health Monitor class.

        Returns:
            bool: True if the resources is healthy, else False.
        """

    def destroy(self) -> bool:
        """
        Destroy a resource.
        
        Returns:
            bool: True if reset was successful, False if resource needs to be re-created.
        """
        pass

    def clean_up(self) -> bool:
        """
        Cleanup the resource when destroying it.
        """
        pass


class Resource(BaseModel):
    api_connection: list[ApiConnection] = None
    func: list[dict[str, Callable]] | list[dict[str,Coroutine]] = None
    gpu_mem: int = None
    thread: int = None
    sys_mem: int = None
    file_path: list[Path] = None

    _use_count: int = PrivateAttr(default=0)

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        return

    async def __anext__(self):
        if self.api_connection:
            yield self.api_connection.pop(0)
        if self.func:
            for func in self.func:
                yield func
        if self.gpu_mem:
            yield self.gpu_mem
        if self.thread:
            yield self.thread
        if self.sys_mem:
            yield self.sys_mem
        if self.file_path:
            yield self.file_path.pop(0)
        raise StopAsyncIteration

    def request(self, requested_items: dict[str, Any]):
        pass

    def create(self) -> T:
        """Create the actual resource."""
        pass

    def validate(self) -> bool:
        """Validate the resources are still healthy."""
        pass

    def cleanup(self):
        """Cleanup the resources when destroying it."""
        pass

    def mark_in_use(self):
        self.state = ResourceState.IN_USE
        self.last_used_at = time.time()
        self.use_count += 1

    def get_resource(self) -> T:
        if self._resource is None:
            self._resource = self.create()
        return self._resource

# Type definitions
T = TypeVar('T')
E = TypeVar('E')

@dataclass
class Either(Generic[E, T]):
    value: T | E
    is_right: bool

    @classmethod
    def right(cls, value: T) -> 'Either[E, T]':
        return cls(value, True)

    @classmethod
    def left(cls, error: E) -> 'Either[E, T]':
        return cls(error, False)

    def map(self, func: Callable[[T], T]) -> 'Either[E, T]':
        if self.is_right:
            return Either.right(func(self.value))
        return self

    def bind(self, func: Callable[[T], 'Either[E, T]']) -> 'Either[E, T]':
        if self.is_right:
            return func(self.value)
        return self


class Converter:

    def __init__(self, resource: Resource, configs: Configs):
        self.configs = configs
        #self.markitdown = MarkItDownAsync()

    async def source(self) -> str:
        pass

    async def drain(self) -> str:
        pass
 
    async def convert(self, url: str) -> DocumentConverterResult:
        pass








class Pool():

    def __init__(self, configs: Configs):
        self.configs = configs
        self.gpu_mem: int = None
        self.sys_mem: int = None
        self.api_connections: int = None
        self.func: dict[str, int] = None
        self.file_paths: list[Path]

        self._initialize_pool()

    def _initialize_pool(self):
        """
        Start up the pool with the initial resources.
        """
        pass

    def put(self, resource: Resource):
        """
        Put a resource back into the pool.
        
        """
        pass

    def dispense(self) -> Generator[Resource]:
        """
        Dispense a resource from the pool.
        It does so by creating a Resource on-the-fly and yield it.

        """
        pass

    @property
    def is_not_full(self) -> bool:
        pass

    @property
    def is_empty(self) -> bool:
        pass



class Pools():

    def __init__(self, configs: Configs):
        self.configs = configs

        self.pool: Pool = self.create_pool(configs)
        self.api_pool: Pool = self.create_pool(configs)
        self.func_pool: Pool = self.create_pool(configs)
        self.gpu_mem_pool: Pool = self.create_pool(configs)
        self.thread_pool: Pool = self.create_pool(configs)
        self.sys_mem_pool: Pool = self.create_pool(configs)
        self.file_path_pool: Pool = self.create_pool(configs)

    def create_pool(self, configs: Configs) -> Pool:
        """
        Instantiate a pool with the given configuration.
        Save for File Paths and API connections, each pool consists of a series of counters. 
           Each counter's upper bound is determined by the maximum values specified in the Configs object. 
           or whatever resources the system has available, whichever is smaller.
           For instance, if the config specifies that the maximum available memory is 1024 MBs, 
           but the system only has 512 MB, then the pool's maximum will only be 512 MBs.
    
        When a Resource object is created, the counter is decremented by the amount of resources 
            that are allocated to the resource.

        When a Resource is consumed by a function, the Pool counter is *not* automatically incremented.
            Instead, the freed resource goes to the Core Resource Manager to be re-allocated to another 
            file in the File Path Queue. If no files in the Queue need that resource, then the resource 
            is returned to the pool, and the Pool's counter is incremented.
            For instance, if the Queue is filled with files that don't require an API connection to convert,
            the connection is returned to the pool and its counter is incremented.

        When a given Pool's counter is at zero, the creation of that Resource will be blocked until resources 
            are returned to the pool, or new Resources are provided by the External Resource Manager, 
            which ever happens sooner. This will have the side-effect of dynamically limiting 
            the number of files that can be processed at any one time.
        
        When a given Pool's counter is at its maximum, returned Resources will be 'thrown away' and the counter 
           remains the same amount. For connections, this means that the connection will be closed (???).

        For API connections, the pool is a traditional connection pool a la MySQL. 
            When a Resource object is created, a connection is allocated to the Resource.
            When the Resource object is consumed, the connection is returned to the pool.
            If the pool is empty, a new connection is requested from the External Resource Manager 
            and added to the pool.

        API connections are refreshed periodically to ensure that they are still valid.

        Args:
            configs (Configs): The configuration for the pool.

        Returns:
           Pool: The instantiated pool.
        """
        return Pool(configs)

    def check_what_resources_the_core_manager_needs(self) -> bool:
        """
        Query the Core Manager to see if it needs any Resources.

        Returns:
            bool: True if any pool is not full or empty, and False if all pools are full.
        """
        pass

    def make_a_resource() -> Generator[Resource]:
        """
        Construct a Resource by taking items from the pools, then yield it.
        When a resource is yielded, it is removed from the pools, and the pool states are updated to reflect
        that the resources have been allocated.

        NOTE: As a guideline, we should over-allocate resources rather than under-allocate.
            Over-allocating is less efficient, but more robust, since a process is less likely to fail
            due to a lack of resources. 

        Returns:
            resource (Resource): A Resource to be taken from the pools.
            Resource is a pydantic base model that can contain any combination of the following:
            - A FilePath pydantic model pointing to an input file. 
                This consists of a path, a CID, and other attributes that are calculated on the fly.
            - References to a series of functions to be executed in order to convert a file.
            - An API connection that is needed by one or more of the functions in order to convert a file.
                As functions are sequential, the API connection is only released after the last function 
                that uses an API connection has been executed.
            - Total system memory (in bytes) necessary to execute the conversion functions.
            - GPU memory (in bytes) necessary to execute the conversion functions. This is primarily
                for conversion functions that rely on local ML models such as Whisper, or a V-LLM.
            - Number of CPU threads necessary to execute the conversion functions.

        Example Return:
            >>> resource = acquire()
            print(resource.model_dump)
            >>> {
                    'path': FilePath('path/to/file'),
                    'api_connection': 1,
                    'threads': 1,
                    'func': [{
                                'step': 1
                                'name': 'open_json',
                                'sys_mem': 512
                            },
                            {
                                'step': 2
                                'name': 'convert_json',
                                'sys_mem': 512,
                                'api_connection': 1,
                                'gpu_mem': 1024
                            },
                            {
                                'step': 3
                                'name': 'save_json',
                                'sys_mem': 256
                            }], 
                    'gpu_mem': 1024, 
                    'sys_mem': 1280
                }
        """
        pass

    def release() -> None:
        pass

    def receive(self, resource: Resource) -> None:
        """
        Receive a resource and put it back into the appropriate pool.

        Args:
            resource (Resource): Resources to be returned to the pool.
            A resource is a pydantic base model that can contain a combination of the following:
            - An API connection
            - References to a Function
            - Available GPU memory (in bytes)
            - CPU Threads
            - Available System memory (in bytes)
        """
        if resource.api_connection:
            self.api_pool.put(resource)
        if resource.func:
            self.func_pool.put(resource)
        if resource.gpu_mem:
            self.gpu_mem_pool.put(resource)
        if resource.thread:
            self.thread_pool.put(resource)
        if resource.sys_mem:
            self.sys_mem_pool.put(resource)

    def send_to_core_manager(self) -> Resource:
        """
        Send a Resource to the Core Manager.
        
        """
        pass


class PoolHealthMonitor:

    def __init__(self, configs: Configs, pools: Pools):
        self.configs = configs
        self.pools = pools

    def check_pool_health(configs) -> None:
        pass

class ExternalResourcesManager():

    def __init__(self, configs: Configs, utility_classes: dict[str, CustomClass]):
        self.configs = configs
        self._resource_holder = []
        self.file_path_manager = utility_classes.pop("file_path_manager")
        self.api_connections = utility_classes.pop("api_connections")
        self.system_resources = utility_classes.pop("system_resources")

    def create_resource(self, paths, system_resources, api_connections) -> AsyncGenerator[Resource]:
        """
        Create a Resource from the available paths, system resources, and API connections.
        
        """
        pass


    def gets_back(self, resource: Resource) -> None:
        """
        
        """
        self._resource_holder.append(resource)


    async def out(self) -> Resource:
        """
        
        """
        pass


    def has_resources(self) -> bool:
        """
        
        """
        pass


    @property
    async def resources(self) -> AsyncGenerator[Resource, None]:
        async for await resource in self.create_resource():
            resource: Resource
            yield resource


class CoreResourceManager():

    def __init__(self, configs: Configs):
        self.configs = configs
        self.available_resources: dict[str, Resource] = None
        self.outputs = None

    def request_a_resource_from_the_pool(self) -> Resource: #Empty resource
        """
        Given the files currently in the FilePathQueue, request a resource from the pool
        
        """
        pass

    def receives(self, resource: Resource) -> None:
        """
        Receive a resource and put it in their respective available resource queue.

        Args:

        """
        pass

    def send_a_resource_back_to_the_pool(self) -> Resource:
        """
        """
        pass

    def send_a_path_to_the_file_queue(self) -> Resource:
        """
        """
        pass

    def send_to_the_converter(self) -> Resource:
        """
        """
        pass

    def returns_that_resource(self, resource: Resource) -> None|Resource:
        """
        """
        pass

    def doesnt_need_a_resource_anymore(self) -> bool:
        """
        Given the files currently in the File Queue, tell the Resource Manager if 
            it doesn't need a resource anymore.

        If the 
        
        """
        pass


def instantiate(this_class: type[CustomClass] = None, with_these = None, and_these = None):
    if this_class is None:
        raise ValueError("You must provide a class to instantiate.")
    if with_these is None:
        raise ValueError("You must provide configs for the class to instantiate.")

    if and_these is None:
        the_instantiated_class = this_class(with_these)
    else:
        the_instantiated_class = this_class(with_these, and_these)

    return the_instantiated_class



class FilePathQueue:

    def __init__(self, configs: Configs):
        self.configs = configs
        self.items = []
        self.lock = asyncio.Lock()
        self.queue = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.queue:
            return await self.queue.get()
        raise StopIteration

    async def add_this(self, item) -> None:
        async with self.lock:
            self.items.append(item)

    async def get_this(self) -> Optional[str]:
        async with self.lock:
            if self.items:
                return self.items.pop(0)
            return None

    async def is_empty(self) -> bool:
        async with self.lock:
            return len(self.items) == 0
        
    async def is_not_full(self) -> bool:
        async with self.lock:
            return len(self.items) <= self.configs.max_queue_size
        
    async def has_enough_resources_to_convert_a_file(self) -> bool:
        pass


from typing import Generator

class Processor:

    def __init__(self, configs, resource):
        self.configs = configs
        self.resource: Resource = resource
        self.either: Either = None

        self.compose = self.compose(configs, resource)

        if self.the_file_needs_async_processing(resource):
            self.processor = self.async_processor(resource)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                yield await self.processor(self.file, self.resource)
            finally:
                raise StopAsyncIteration

    def processor(file: Path, resource: Resource) -> Generator[Resource]:
        """
        Compose resources into a processor chain.
        A processor chain is a series of functions that take a Resource and return left-overs as they go.
        A chain requires a Resource and a file to process:

           - file: Path
           - functions: dict[str, Callable]
           - API connections: list[ApiConnection]R
           - System Resources: list[SystemResource]
        
        """

    async def async_processor(file, resource: Resource) -> AsyncGenerator[Resource]:
        """
        
        
        """
        pass

    def compose(resource: Resource) -> Either:
        """
        Take functions from a resourec and compose them into a chain.

        The chain is a list of functions and/or coroutines that will be executed in order.
        """
        pass

    def the_file_needs_async_processing(file):
        """
        
        """
        pass

    def returns_leftover_resources() -> Resource | None:
        """
        If there are any leftover resources, return them back to the Core Manager.

        """
        pass

def figure_out_what_resources_we_need_for_this(file):
    pass





async def main():

    logger.info("Begin __main__")

    next_step("Step 1: Load in the configs.")
    if _program_was_started_from_command_line():
        configs = run_with_argparse()
    else:
        parser = ConfigParser()
        configs = parser.load_and_parse_configs_file()


    next_step("Step 2: Create and start the File Paths Manager")
    file_paths_manager = instantiate(this_class=FilePathsManager, with_these=configs)


    next_step("Step 3: Create and start the LLM API.")
    llm_api_manager = instantiate(this_class=LlmApiManager, with_these=configs)


    next_step("Step 4: Create and start the System Resources Manager.")
    system_resources_manager = instantiate(this_class=SystemResourcesManager, with_these=configs)


    next_step("Step 5: Create and start the External Resource Manager.")
    classes = {
        'file_paths_manager':file_paths_manager, 
        'llm_api_manager':llm_api_manager, 
        'system_resources_manager':system_resources_manager
    }
    # NOTE This should run in its own thread.
    the_erm = instantiate(this_class=ExternalResourcesManager, with_these=configs, and_these=classes)


    next_step("Step 6: Create and start the Core Resource Manager.")
    # NOTE This should run in its own thread.
    the_core = instantiate(this_class=CoreResourceManager, with_these=configs)


    next_step("Step 7: Create and start the Pool Health Monitor.")
    pool_health_monitor = instantiate(this_class=PoolHealthMonitor, with_these=configs, and_these=[the_pools])


    next_step("Step 8: Create and start the Pools.")
    # NOTE This should run in its own thread.
    the_pools = instantiate(this_class=Pools, with_these=configs, and_these=pool_health_monitor)


    next_step("Step 9: Start the main loop.")
    while True:

        next_step("Step 10: Generate a resource.")
        async for this_resource in the_erm.resources:

            next_step("Step 11: Check if the pool needs a resource.")
            if the_pools.need_a_resource():
                next_step("Step 12a: Send a resource to the Pools if the Pool need it.")
                the_pools.receive(this_resource)
            else:
                next_step("Step 12b: Return a resource to the ExternalResourcesManager if the Pools don't need it.")
                the_erm.gets_back(this_resource)

            next_step("Step 13: Send a resource to the CoreResourceManager if the CoreResourceManager needs it.")
            if the_core.needs_a_resource():
                the_pools_resource = the_pools.make_a_resource()
                the_core.receives(the_pools_resource)

            next_step("Step 14: Return a resource to the ExternalResourcesManager if the CoreResourceManager doesn't need it.")
            if the_core.doesnt_need_a_resource_anymore():
                that_resource = the_core.returns_that_resource()
                the_erm.gets_back(that_resource)

        # NOTE: We can run sync functions in an async loop, but not vice-versa.
        for file, resources in the_core:
            next_step("Step 15: Instantiate a queue of files and fill it up.")
            the_queue = FilePathQueue(configs)
            if await the_queue.is_not_full():
                await the_queue.add_this(file)

            async for file in the_queue:
                next_step("Step 16: Figure out what resources we need, then allocate them.")
                what_we_need = figure_out_what_resources_we_need_for_this(file)
                allocated_resources = the_core.gives_us(what_we_need, from_these=resources)

                next_step("Step 17: Actually convert the file.")
                # NOTE This should run in its own thread.
                this_specific_processor = Processor(file, configs, allocated_resources)
                try:
                    next_step("Step 18: As we run through the conversion steps, give resources back to the Core Resource Manager.")
                    async for this_used_resource in this_specific_processor:
                        the_core.receives(this_used_resource)
                except:
                    ("Step 18b: Return remaining resources to the pool on failure.")
                    # NOTE: We don't care why it failed, we just want to make sure we get our resources back.
                    # It's all about keeping everything going!
                    the_leftover_resources = await this_specific_processor.returns_leftover_resources()
                    the_core.receives(the_leftover_resources)
                finally:
                    next_step("Step 19: Return all the left over resources to the pool.")
                    the_leftover_resources = await this_specific_processor.returns_leftover_resources()
                    the_core.receives(the_leftover_resources)
                    the_queue.removes_this(file)


    # # Load in the config file.
    # configs = Configs()

    # # Get the files/URLs to process and put the paths to them into a duckdb database
    # configs.input_db = duckdb.connect('input.db')
    # configs.output_db = duckdb.connect('output.db')

    # configs.input_db.create_function(
    #     'make_sha256_hash', make_sha256_hash, 'data', return_type=VARCHAR
    # )

    # configs.input_db.execute(
    #     "CREATE TABLE IF NOT EXISTS input (file_path VARCHAR, uuid VARCHAR)"
    # )
    # file_paths = [
    #     str(path) for path in configs.paths.INPUT_DIR.glob("**/*") 
    #     if path.is_file()
    # ]

    # # Split the file_paths into chunks based on the number of workers
    # for file_path in file_paths:
    #     configs.input_db.execute(
    #         "INSERT file_path, uuid INTO input VALUES (?), (make_sha256_hash(?)", [file_path, file_path]
    #     )

    # # Divide the data based on their mime-type

    # Assign the data 


    #mimetypes.guess_type(url)

    #logger.info("Insert program logic here...")
    #logger.info("End __main__")

    sys.exit(0)


if __name__ == "__main__":

    import os
    base_name = os.path.basename(__file__) 
    program_name = os.path.split(os.path.split(__file__)[0])[1] if base_name != "main.py" else os.path.splitext(base_name)[0] 
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"'{program_name}' program stopped.")
