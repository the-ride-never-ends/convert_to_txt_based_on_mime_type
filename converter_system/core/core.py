import asyncio
from collections import Counter
import concurrent.futures as cf
import logging
from queue import Queue
import time
import threading
from typing import Any, AsyncGenerator, AsyncIterator, Generator, Iterator, Optional, TypeVar


from pydantic_models.configs import Configs
from pydantic_models.resource.resource import Resource
from utils.common.monads.async_ import Async
from logger.logger import Logger

T = TypeVar('T')


logger = Logger(__name__)



@asyncio.coroutine
def gather(l: Any) -> Generator:
    return (yield from asyncio.gather(*l, return_exceptions=True))


class AsyncStreamProcessor:
    """Processes a stream of resources in parallel with controlled concurrency"""
    
    def __init__(self, max_workers: int, queue_size: int = 100):
        self.executor = cf.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='stream_worker'
        )
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.processing = set()
        self.done = asyncio.Event()

    async def feed_queue(self, iterator: Iterator[T]) -> None:
        """Feed items from iterator into the async queue"""
        try:
            for item in iterator:
                await self.queue.put(item)
        finally:
            await self.queue.put(None)  # Sentinel to signal end of stream

    async def process_stream(self, 
                           source: Iterator[T],
                           processor: callable) -> AsyncGenerator[Counter, None]:
        """Process items from the stream with parallel execution"""
        
        # Start queue feeder in the background
        feeder = asyncio.create_task(self.feed_queue(source))
        
        try:
            while True:
                # Get next item from queue
                item = await self.queue.get()
                if item is None:
                    break

                # Submit work to thread pool
                future = self.executor.submit(processor, item)
                self.processing.add(future)
                
                # Clean up completed futures and yield results
                done, self.processing = cf.wait(
                    self.processing,
                    timeout=0,
                    return_when=cf.FIRST_COMPLETED
                )
                
                for future in done:
                    try:
                        result = future.result()
                        yield result
                    except Exception as e:
                        # Handle or log errors as needed
                        print(f"Error processing item: {e}")

            # Wait for remaining tasks
            if self.processing:
                done, _ = cf.wait(self.processing)
                for future in done:
                    try:
                        result = future.result()
                        yield result
                    except Exception as e:
                        print(f"Error processing item: {e}")
                        
        finally:
            self.done.set()
            feeder.cancel()

class ResourceGenerator:
    """Simulates a source of resources being generated"""
    
    def __init__(self, total_resources: int, delay: float = 0.1):
        self.total = total_resources
        self.delay = delay
        self.queue = Queue()
        self.thread = None
        
    def generate_resources(self):
        """Generate resources in a separate thread"""
        for i in range(self.total):
            time.sleep(self.delay)  # Simulate work
            resource = Resource(thread=i)
            self.queue.put(resource)
    
    def start(self):
        """Start the resource generation thread"""
        self.thread = threading.Thread(target=self.generate_resources)
        self.thread.start()
    
    def __iter__(self):
        """Make this an iterator that yields resources as they're generated"""
        remaining = self.total
        while remaining > 0:
            resource = self.queue.get()
            yield resource
            remaining -= 1

class Core:
    def __init__(self, configs: Configs):
        self.loop = asyncio.get_event_loop()
        self.semaphore = asyncio.Semaphore(configs.concurrency_limit)
        self.stream_processor = AsyncStreamProcessor(
            max_workers=configs.concurrency_limit # TODO
        )

    async def process_resource(self, resource: Resource) -> Counter:
        """Process a single resource"""
        async with self.semaphore:
            pipeline = (
                Async(resource.load())
                >> gather
                >> (lambda x: filter(lambda: not isinstance(x, Exception), x))
                >> (lambda x: map(lambda: resource.convert(x), x))
                >> gather
                >> (lambda x: filter(lambda s: not isinstance(s, Exception), x))
                >> (lambda x: map(lambda: resource.save(x), x))
                >> Counter
            )
            return await pipeline.future

    async def process_stream(self, 
                           resource_stream: Iterator[Resource]) -> AsyncIterator[Counter]:
        """Process a stream of resources in parallel"""
        async for result in self.stream_processor.process_stream(
            resource_stream,
            lambda r: self.loop.run_until_complete(self.process_resource(r))
        ):
            yield result

# Usage example:
if __name__ == "__main__":
    configs = Configs(concurrency_limit=10)
    core = Core(configs)
    
    # Create resource generator
    generator = ResourceGenerator(total_resources=100)
    generator.start()
    
    async def main():
        results = []
        async for result in core.process_stream(generator):
            results.append(result)
            print(f"Processed resource, total complete: {len(results)}")
    
    # Run the pipeline
    core.loop.run_until_complete(main())