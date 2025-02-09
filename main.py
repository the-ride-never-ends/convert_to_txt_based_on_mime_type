import asyncio
import hashlib
import logging
import os
import mimetypes
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Coroutine, IO, Iterable, Never, Optional, TypeVar
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
from pydantic import BaseModel, Field, HttpUrl, EmailStr


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


class Converter:

    def __init__(self, configs: Configs):
        self.configs = configs
        #self.markitdown = MarkItDownAsync()

    async def source(self) -> str:
        pass

    async def drain(self) -> str:
        pass
 
    async def convert(self, url: str) -> DocumentConverterResult:
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


CustomClass = TypeVar("CustomClass")

class ApiConnection(BaseModel):
    pass

class SystemResource(BaseModel):
    pass


class LlmApiManager():
    
    def __init__(self, configs: Configs):
        self.configs = configs

    def out(self) -> list[ApiConnection]:
        pass


class SystemResourcesManager():

    def __init__(self, configs: Configs):
        self.configs = configs

    def out(self) -> list[SystemResource]:
        pass


class Resource(BaseModel):
    api_connection: ApiConnection = None
    func: Callable | Coroutine = None
    gpu_mem: int = None
    thread: int = None
    sys_mem: int = None
    file_path: Path = None


class Pool():

    def __init__(self, configs: Configs):
        self.configs = configs

    def put(self, resource: Resource):
        pass

    def dispense(self) -> Resource:
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
        Instead of having the objects themselves be the pool, the pool consists of references to the objects.
        For instance, if the pool is for API connections, it will consist of references 
            to the API connections, not the API connections themselves.
        
        Args:
            configs (Configs): The configuration for the pool.

        Returns:
           Pool: The instantiated pool.
        """
        return Pool(configs)

    def need_a_resource(self) -> bool:
        """
        Check the pools to see if they need any resources.
        
        Returns:
            bool: True if any pool is not full or empty, and False if all pools are full.
        """
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

    def send(self) -> Resource:
        pass


class PoolHealthMonitor:

    def __init__(self, configs: Configs, pools: Pools):
        self.configs = configs
        self.pools = pools.copy()

    def check_pool_health(configs) -> None:
        pass

class ExternalResourcesManager():

    def __init__(self, configs: Configs, utility_classes: list[CustomClass]):
        self.configs = configs
        self._resource_holder = []

    def create(self, paths, system_resources, api_connections) -> list[Resource]:
        pass

    def gets_back(self, resource: Resource) -> None:
        self._resource_holder.append(resource)

    async def out(self) -> Resource:
        pass

    def has_resources(self) -> bool:
        pass

    @property
    async def resources(self) -> AsyncGenerator[Resource, None]:
        async for await resource in self.out():
            resource: Resource
            yield resource


class CoreResourceManager():

    def __init__(self, configs: Configs):
        self.configs = configs

    def needs_a_resource(self) -> bool:
        pass

    def receives(self, resource: Resource) -> None:
        pass

    def returns_that_resource(self, resource: Resource) -> None:
        pass

    def doesnt_need_a_resource_anymore(self) -> None:
        pass


class FilePathQueue():

    def __init__(self):
        self.queue = asyncio.Queue()


class ConsumableList:

    def __init__(self):
        self.items = []
        self.lock = asyncio.Lock()

    async def add(self, item):
        async with self.lock:
            self.items.append(item)

    async def get(self):
        async with self.lock:
            if self.items:
                return self.items.pop(0)
            return None

    async def is_empty(self):
        async with self.lock:
            return len(self.items) == 0


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
    classes = [file_paths_manager, llm_api_manager, system_resources_manager]
    the_erm = instantiate(this_class=ExternalResourcesManager, with_these=configs, and_these=classes)


    next_step("Step 6: Create and start the Core Resource Manager.")
    the_core_resource_manager = instantiate(this_class=CoreResourceManager, with_these=configs)


    next_step("Step 7: Create and start the Pool Health Monitor.")
    pool_health_monitor = instantiate(this_class=PoolHealthMonitor, with_these=configs, and_these=[the_pools])


    next_step("Step 8: Create and start the Pools.")
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
            if the_core_resource_manager.needs_a_resource():
                the_core_resource_manager.receives(this_resource)


            next_step("Step 14: Return a resource to the ExternalResourcesManager if the CoreResourceManager doesn't need it.")
            if the_core_resource_manager.doesnt_need_a_resource_anymore():
                that_resource = the_core_resource_manager.returns_that_resource()
                the_erm.gets_back(that_resource)


        # NOTE: We can run sync functions in an async loop, but not vice-versa.
        for file, resources in the_core_resource_manager:
            next_step("Step 15: Instantiate a queue of files and fill it up.")
            the_queue = FilePathQueue(configs)
            if await the_queue.is_not_full():
                await the_queue.add_this(file)

            async for file in the_queue:
                next_step("Step 16: Figure out what resources we need, then allocate them.")
                what_we_need = figure_out_what_resources_we_need_for_this(file)
                allocated_resources = the_core_resource_manager.gives_us(what_we_need, from_these=resources)

                next_step("Step 17: Actually convert the file.")
                processor = Processor(file, configs, allocated_resources)
                try:
                    async for this_used_resource in processor:
                        the_core_resource_manager.receives(this_used_resource)
                except:
                    # NOTE: We don't care why it failed, we just want to make sure we get our resources back.
                    # It's all about keeping everything going!
                    the_leftover_resources = await processor.returns_leftover_resources()
                    the_core_resource_manager.receives(the_leftover_resources)
                finally:
                    the_queue.remove_this(file)


class Processor:

    def __init__(self, file, configs, resources):
        self.configs = configs
        self.resources = resources

        if self.the_file_needs_async_processing(file):
            self.processor = self.async_processor(file, resources)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                yield await self.processor(self.file, self.resources)
            except StopAsyncIteration:
                raise StopAsyncIteration

    def processor(file, resources):
        pass

    async def async_processor(file, resources):
        pass

    def the_file_needs_async_processing(file):
        pass

    def returns_leftover_resources():
        pass

def figure_out_what_resources_we_need_for_this(file):
    pass

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
