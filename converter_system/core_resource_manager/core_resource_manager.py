



from functools import partial
from typing import Any, Callable, Coroutine, Generator, Optional


import duckdb


from pydantic_models.configs import Configs
from pydantic_models.resource.resource import Resource
from utils.errors.error_on_wrong_value import error_on_wrong_value

class CoreResourceManager:

    def __init__(self):
        pass

    async def allocate_current_resources_to_queue():
        pass

    async def request_resources_from_pool_resource_manager():
        pass



import asyncio
import concurrent.futures as cf
import logging
import traceback
logger=logging.getLogger('monadic_cf')

class TaskError(Exception):
    pass

def check_result(future: cf.Future, chained: Optional[cf.Future] = None):
    if future.exception():
        logger.error(f'Exception on future {future.exception()}')
        if chained:
            chained.set_exception(future.exception())
        raise TaskError()
    elif future.cancelled():
        logger.debug(f'{future} was cancelled.')
        if chained:
            chained.cancel()
        raise TaskError()
    else:
        return future.result()

def pass_result(resolved: cf.Future, unresolved: cf.Future):
    if resolved.exception():
        unresolved.set_exception(resolved.exception())
    elif resolved.cancelled():
        unresolved.cancel()
    else:
        unresolved.set_result(resolved.result())


@asyncio.coroutine
def gather(l: Any) -> Generator:
    return (yield from asyncio.gather(*l, return_exceptions=True))


from collections import Counter
from converter_system.core_error_manager.core_error_manager import CoreErrorManager 


@asyncio.coroutine
def load(resource: Resource) -> Resource:
    return resource.load()

@asyncio.coroutine
def convert(resource: Resource) -> Resource:
    return resource.convert()

@asyncio.coroutine
def save(resource: Resource) -> Resource:
    return resource.save()


def filter_errors(resource: Resource):
    return filter(
        lambda s: not isinstance(s, Exception), resource
    )

class Core():

    def __init__(self, configs: Configs):

        self.loop_run_until_complete = asyncio.get_event_loop().run_until_complete
        self.available_threads = []

        # Never have more concurrent tasks than the number of available threads.
        self.semaphore = asyncio.Semaphore(configs.concurrency_limit)

        self.core_db = duckdb.connect(':memory:')
        self.error_manager = CoreErrorManager(configs)

    async def run(self, resource: Resource):

        def log_results(resource):
            return partial(
                map, lambda: self.error_manager.log(resource)
            )


        async with self.semaphore:
            pipeline = AsyncCF(
                partial(map, load(resource)), # Load 
                executor=self.available_threads
                ) >> gather >> filter_errors >> partial(
                    map, convert(resource) # Convert
                ) >> gather >> filter_errors >> partial(
                    map, save(resource) # Save
                ) >> gather >> filter_errors >> Counter

        return self.loop_run_until_complete(pipeline.future)




class AsyncCF(object):
    """
    Source: http://zderadicka.eu/functional-fun-with-asyncio-and-monads/
    """
    _executor = cf.ThreadPoolExecutor(4)

    def __init__(self, work: Any, *args, **kwargs):
        try:
            self._executor = cf.ThreadPoolExecutor(kwargs.pop('executor'))
        except KeyError:
            raise ValueError('Supply executor')

        if isinstance(work, cf.Future):
            self._future = work
        elif callable(work):
            self._future = self._executor.submit(work, *args, **kwargs)
        else:
            self._future = self._executor.submit(lambda: work)

        self._chained = None

    def bind(self, next_work) -> None:

        if not callable(next_work):
            raise ValueError('Expected callable')

        def resolved(f):
            try:
                res = check_result(f, self._chained)
            except TaskError:
                return
            t = self._executor.submit(next_work, res)
            t.add_done_callback(lambda f: pass_result(f, new_future))

        new_future = cf.Future()
        next_async = AsyncCF(new_future, executor=self._executor)
        self._chained = new_future
        self._future.add_done_callback(resolved)
        return next_async

    def __rshift__(self, other):
        return self.bind(other)

    def __or__(self, other):
        return self.bind(other)

    @property
    def future(self):
        return self._future

    @property
    def result(self):
        if self._future.exception():
            raise self._future.exception()
        return self._future.result()

    def wait_finished(self):
        cf.wait([self._future])
        return self.result






# class Async(object):

#     def __init__(self, work: Any, *args, **kwargs):
    
#         if isinstance(work, asyncio.Future):
#             self._future=work
#         elif asyncio.iscoroutine(work):
#             self._future = asyncio.create_task(work)
#         elif callable(work):
#             self._future=asyncio.async(assure_coro_fn(work)(*args, **kwargs))
#         else:
#             self._future=asyncio.create_task(asyncio.coroutine(lambda: work)())
#         self._chained = None

#     def bind(self, next_work):
#         next_work = assure_coro_fn(next_work)
#         def resolved(f):
#             try:
#                 res=check_result(f, self._chained)
#             except TaskError:
#                 return
#             t = asyncio.create_task(next_work(res))
#             t.add_done_callback(lambda f: pass_result(f, new_future))

#         new_future = asyncio.Future()
#         next_async = Async(new_future)
#         self._chained=  new_future
#         self._future.add_done_callback(resolved)
#         return next_async

#     def __rshift__(self, other):
#         return self.bind(other)

#     def __or__(self, other):
#         return self.bind(other)

#     @property
#     def future(self):
#         return self._future