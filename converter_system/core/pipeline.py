import asyncio
from functools import wraps
import logging
from typing import Any, Coroutine, Generator, Optional, TypeVar

from utils.common.monads.async_ import Async
from logger.logger import Logger

Resource = TypeVar('Resource')

logger = Logger(__name__)

@asyncio.coroutine
def log_errors(resource: Resource) -> Optional[Resource]:
    if isinstance(resource, Exception):
        logger.error(f"Error in Pipeline: {resource}")
        return None
    return resource 

@asyncio.coroutine
def start_pipeline(resource) -> Resource:
    return resource

async def load(resource) -> Optional[Resource]:
    return await resource.load()

async def convert(resource) -> Optional[Resource]:
    return await resource.convert()

async def save(resource) -> Optional[Resource]:
    return await resource.save()

async def catch_exceptions(resource: Any) -> Optional[Resource]:
    return lambda x: x if isinstance(x, Exception) else resource

async def raise_exceptions(resource: Any) -> Optional[Resource]:
    if isinstance(resource, Exception):
        raise resource
    return resource

async def filter_exceptions(resource: Any) -> Optional[Resource]:
    return lambda x: filter(lambda: not isinstance(x, Exception), resource)

@asyncio.coroutine
def gather(l: Any) -> Generator:
    return (yield from asyncio.gather(*l, return_exceptions=True))

async def pass_fail(resource: Optional[Resource]) -> None:
    return lambda: (
        logger.info("Conversion succeeded."), resource
    ) if resource is not None else (
        logger.error("Conversion failed."), resource
    )


class Pipeline(object):

    def __init__(self, resource):

        self.resource = resource
        self.start_pipeline = None
        self.pipeline = None

        self.start_pipeline: Coroutine = start_pipeline(resource)
        self.pipeline = Async(
            self.start_pipeline
            ) >> gather >> log_errors >> (
                lambda resource: load(resource)
            ) >> gather >> log_errors >> (
                lambda resource: convert(resource)
            ) >> gather >> log_errors >> (
                lambda resource: save(resource)
            ) >> gather >> pass_fail

    def run(self):
        return self.pipeline.future