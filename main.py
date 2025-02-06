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
from typing import Any, Coroutine, IO, Iterable, Optional
from urllib.parse import urlparse
from uuid import UUID


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
        self.markitdown = MarkItDownAsync()

    async def source(self) -> str:
        pass

    async def drain(self) -> str:
        pass
 
    async def convert(self, url: str) -> DocumentConverterResult:
        pass




class MarkItDownAsync(MarkItDown):
    """
    A modified version of MarkItDown that supports asynchronous use.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asyncio_lock = asyncio.Lock()
        self._concurrency_limit: int = kwargs.get("concurrency_limit", 5)
        self._aiohttp_session = None
        self._browser: Browser = None
        self._playwright_dom_manipulation_ruleset: dict[str, Coroutine | list[Coroutine]] = None
        self._we_have_async: bool = None

        try: # Importing these should tell is whether or not we have async functionality.
            import playwright
            import aiohttp
            self._we_have_async = True
        except:
            self._we_have_async = False

        if self._we_have_async:

            aiohttp_session: aiohttp.ClientSession = kwargs.get("aiohttp_session", None)
            self._aiohttp_session = aiohttp.ClientSession() if aiohttp_session is None else aiohttp_session

            playwright_context_manager: PlaywrightContextManager = kwargs.get("playwright_context_manager", None)

            if playwright_context_manager is None:
                print("Warning: No Playwright context manager provided. Loading URLs with Playwright will be disabled.")
            else:
                playwright_configs: Configs = kwargs.get("playwright_configs", None)
                if playwright_configs is not None:
                    self._browser = playwright_context_manager.chromium.launch(**playwright_configs)
                else:
                    self._browser = playwright_context_manager.chromium.launch(headless=True)

    @classmethod
    def enter(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        return instance

    async def exit(self):
        if self._aiohttp_session is not None:
            await self._aiohttp_session.close()
        if self._browser is not None:
            await self._browser.close()

    def raise_runtime_error_if_no_async(self):
        if not self._we_have_async:
            raise RuntimeError("Async functionality is not available. Please install the required dependencies.")

    async def async_convert(self, 
                      source: str | aiohttp.ClientResponse | Path | requests.Response, 
                      **kwargs: Any
                    ) -> DocumentConverterResult:
        """
        Args:
            source: can be a string representing one of the following:
                - a local path to a file, either as string or a pathlib.Path object
                - a url string
                - a requests.response object
                - a aiohttp.ClientResponse object
            extension: specifies the file extension to use when interpreting the file. 
                If None, infer from source (path, uri, content-type, etc.)

        Returns
           DocumentConverterResult: A pydantic model containing the result of converting a document to text.
        """
        self.raise_runtime_error_if_no_async()

        # Local path or url
        if isinstance(source, str):
            if (
                source.startswith("http://")
                or source.startswith("https://")
                or source.startswith("file://")
            ):
                if self._aiohttp_session is None:
                    return self.convert_url(source, **kwargs)
                else:
                    return await self.async_convert_url(source, **kwargs)
            else:
                return self.convert_local(source, **kwargs)

        # Request response
        elif isinstance(source, requests.Response):
            return self.convert_response(source, **kwargs)
        elif isinstance(source, Path):
            return self.convert_local(source, **kwargs)
        elif isinstance(source, aiohttp.ClientResponse):
            return self.convert_aiohttp_response(source, **kwargs)
        elif isinstance(source, PlaywrightResponse) and self._browser is not None:
            return self.convert_playwright_response(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")


    async def async_convert_url(
        self, url: str, **kwargs: Any
    ) -> DocumentConverterResult:
        self.raise_runtime_error_if_no_async()

        # Send a HTTP request to the URL
        with self._aiohttp_session.get(url, stream=True) as response:
            response: aiohttp.ClientResponse
            response.raise_for_status()
            text_response = await response.text()
        try:
            # Check if we need to render the page with Playwright.
            if self._figure_out_whether_we_need_to_render_this_url_with_playwright(text_response) and self._browser is not None:
                return await self._convert_url_with_playwright(url, **kwargs)
            else:
                return self.convert_aiohttp_response(response, **kwargs)
        except: # If we can't parse the HTML, then it's probably not a webpage.
            return self.convert_aiohttp_response(response, **kwargs)


    async def _convert_url_with_playwright(self, url: str, **kwargs: Any) -> DocumentConverterResult:
        # NOTE Since we always call aiohttp before right this, we can assume that we'll never get a 404 here.
        page: Page = await self._browser.new_page()
        _page = await page.goto(url)
        # if self._playwright_dom_manipulation_ruleset:
        #     if url in self._playwright_dom_manipulation_ruleset:
        #         for coroutine in self._playwright_dom_manipulation_ruleset[url]:
        #             await coroutine(page)
        if _page is not None:
            await page.wait_for_load_state('networkidle')
            response = await _page.request.response()
            try:
                return await self.convert_playwright_response(response, page, **kwargs)
            finally:
                await page.close()
        else:
            return None

    # TODO Make this more robust.
    def _figure_out_whether_we_need_to_render_this_url_with_playwright(self, html: str) -> bool:
        # Count how many div HTML tags there are.
        num_of_divs = html.lower().count('<div')

        # Check if there's a lot of camel case words in the webpage.
        num_of_camel_case_words = _count_number_of_camel_case_words(html)

        return True if num_of_camel_case_words >= num_of_divs else False


    def _read_the_extension_from_the_path(self, extensions: list[str], url: str) -> None:
        base, ext = os.path.splitext(urlparse(url).path)
        self._append_ext(extensions, ext)


    async def _guess_from_the_mimetype(self, 
                        extensions: list[str], 
                        response: aiohttp.ClientResponse | PlaywrightResponse
                        ) -> None:
        if isinstance(response, PlaywrightResponse):
            content_type = response.content_type.split(";")[0] if response.content_type else ""
        else:
            content_type = response.headers.get("content-type", "").split(";")[0]
        self._append_ext(extensions, mimetypes.guess_extension(content_type))


    async def _read_the_content_disposition_if_there_is_one(
                        self,
                        extensions: list[str], 
                        response: aiohttp.ClientResponse | PlaywrightResponse, 
                        ) -> None:
        # Read the content disposition if there is one
        # Get the headers from the page
        if isinstance(response, PlaywrightResponse):
            content_disposition = response.content_disposition if response.content_disposition else ""
        else:
            content_disposition = response.headers.get("content-disposition", "")
        m = re.search(r"filename=([^;]+)", content_disposition)
        if m:
            base, ext = os.path.splitext(m.group(1).strip("\"'"))
            self._append_ext(extensions, ext)


    async def _download_and_convert_the_file(
                        self, 
                        fh: IO[bytes],
                        extensions: list[str], 
                        temp_path: str,
                        response: aiohttp.ClientResponse | PlaywrightResponse,
                        **kwargs: Any
                        ) -> DocumentConverterResult:
        chunk_size = 512  # Define chunk size (1/2 KB per write)
        if isinstance(response, PlaywrightResponse):
            # Get the response body
            bytes_ = await response.body()

            # Download the file
            for chunk in range(0, len(bytes_), chunk_size):
                fh.write(bytes_[chunk:chunk+chunk_size])
        else:
            # Download the file
            async for chunk in response.content.iter_chunked(chunk_size):
                fh.write(chunk)
            await response.release()
        fh.close()

        # Use puremagic to check for more extension options
        for g in self._guess_ext_magic(temp_path):
            self._append_ext(extensions, g)

        return self._convert(temp_path, extensions, url=response.url, **kwargs)

    def prepare_a_list_of_extensions_to_try_in_order_of_priority(self,  kwargs: Any) -> list[str]:
        # Prepare a list of extensions to try (in order of priority)
        ext = kwargs.get("file_extension")
        return [ext] if ext is not None else []

    async def convert_playwright_response(self,
        response: PlaywrightResponse, **kwargs: Any
    ) -> DocumentConverterResult:
        self.raise_runtime_error_if_no_async()

        extensions = self.prepare_a_list_of_extensions_to_try_in_order_of_priority(kwargs)

        await self._guess_from_the_mimetype(extensions, response)
        await self._read_the_content_disposition_if_there_is_one(extensions, response)
        self._read_the_extension_from_the_path(extensions, response.url)

        # Save the file locally to a temporary file. It will be deleted before this method exits
        handle, temp_path = tempfile.mkstemp()
        fh = os.fdopen(handle, "wb")
        result = None
        try:
            self._download_and_convert_the_file(                        
                fh, extensions, temp_path, response, **kwargs
            )
        finally:
            self._clean_up(fh, response, temp_path, response)
        return result
    
    def _clean_up(self, fh: IO[bytes], temp_path: str, response: aiohttp.ClientResponse | PlaywrightResponse) -> None:
        try:
            fh.close()
        except Exception:
            pass
        try:
            response.close()
        except:
            pass
        os.unlink(temp_path)

    async def convert_aiohttp_response(
        self, response: aiohttp.ClientResponse, **kwargs: Any
    ) -> DocumentConverterResult:  # TODO fix kwargs type
        self.raise_runtime_error_if_no_async()

        extensions = self.prepare_a_list_of_extensions_to_try_in_order_of_priority(kwargs)

        await self._guess_from_the_mimetype(extensions, response)
        await self._read_the_content_disposition_if_there_is_one(extensions, response)
        self._read_the_extension_from_the_path(extensions, str(response.url))

        # Save the file locally to a temporary file. It will be deleted before this method exits
        handle, temp_path = tempfile.mkstemp()
        fh = os.fdopen(handle, "wb")
        result = None
        try:
            self._download_and_convert_the_file(                        
                fh, extensions, temp_path, response, **kwargs
            )
        # Clean up
        finally:
            self._clean_up(fh, response, temp_path, response)
        return result


from utils.logger.logger import Logger
logger = Logger(__name__)

def make_sha256_hash(data: Any) -> str:
    return hashlib.sha256(str(data).encode())

def make_hash_tree(*args: Iterable[Any]):
    hashed_objects = [
        make_sha256_hash(arg) for arg in args
    ]


async def main():

    logger.info("Begin __main__")

    # Load in the config file.
    configs = Configs()

    # Get the files/URLs to process and put the paths to them into a duckdb database
    configs.input_db = duckdb.connect('input.db')
    configs.output_db = duckdb.connect('output.db')

    configs.input_db.execute(
        "CREATE TABLE IF NOT EXISTS input (file_path VARCHAR, uuid VARCHAR)"
    )
    file_paths = [
        str(path) for path in configs.paths.INPUT_DIR.glob("**/*") 
        if path.is_file()
    ]

    # Split the file_paths into chunks based on the number of workers
    

    for file_path in file_paths:
        configs.input_db.execute(
            "INSERT file_path, uuid INTO input VALUES (?), (?)", [file_path, ]
        )

    # Divide the data based on their mime-type

    # Assign the data 


    #mimetypes.guess_type(url)

    logger.info("Insert program logic here...")
    logger.info("End __main__")

    sys.exit(0)


if __name__ == "__main__":
    import os
    base_name = os.path.basename(__file__) 
    program_name = os.path.split(os.path.split(__file__)[0])[1] if base_name != "main.py" else os.path.splitext(base_name)[0] 
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"'{program_name}' program stopped.")


