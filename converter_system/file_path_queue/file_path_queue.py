
import asyncio
from asyncio import AbstractEventLoop
import concurrent.futures
import itertools
from functools import singledispatch
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union
from pydantic import BaseModel


from pydantic_models.resource.resource import Resource
from pydantic_models.configs import Configs


class Resource(BaseModel):
    pipeline: Any
    prefer: str = "processor"

class ProcessInput(BaseModel):
    resource: Resource
    prefer: str = "processor"

class ThreadInput(BaseModel):
    resource: Resource
    prefer: str = "thread"


class QueueManager:
    def __init__(self, configs: Configs):
        self.batch_size = configs.batch_size
        self.process_queue: asyncio.Queue[ProcessInput] = asyncio.Queue(maxsize=configs.max_queue_size)
        self.thread_queue: asyncio.Queue[ThreadInput] = asyncio.Queue(maxsize=configs.max_queue_size)
        self.loop = asyncio.get_event_loop()

    async def core_resource_manager_interface(self):
        while True:
            item_count = sum(self.process_queue.qsize(), self.thread_queue.qsize())
            if item_count < self.batch_size:
                pass


    async def add_resource(self, resource: Resource):
        if resource.prefer == "thread":
            await self.thread_queue.get(
                ThreadInput(resource=resource)
            )
        else:
            await self.process_queue.get(
                ProcessInput(resource=resource)
            )

    def get_queues(self) -> Tuple[asyncio.Queue[ProcessInput], asyncio.Queue[ThreadInput]]:
        return self.process_queue, self.thread_queue
