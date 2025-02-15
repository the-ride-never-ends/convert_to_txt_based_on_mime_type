import asyncio
import logging
from typing import Any

from logger.logger import Logger

logger = Logger(__name__)

class TaskError(Exception):
    pass

def check_result(f: asyncio.Future, chained: asyncio.Future = None):
    if f.exception():
        if chained:
            chained.set_exception(f.exception())
        raise TaskError()
    elif f.cancelled():
        logger.debug(f'{f} was cancelled')
        if chained:
            chained.cancel()
        raise TaskError()
    else:
        return f.result()

def pass_result(resolved: asyncio.Future, unresolved: asyncio.Future):
    if resolved.exception():
        unresolved.set_exception(
            resolved.exception()
        )
    elif resolved.cancelled():
        unresolved.cancel()
    else:
        unresolved.set_result(
            resolved.result()
        )

@asyncio.coroutine
def ensure_this_is_a_coroutine(fn_or_coro):
    """
    Coerce a function into being a coroutine.
    """
    if asyncio.iscoroutinefunction(fn_or_coro):
        return fn_or_coro

    elif callable(fn_or_coro):
        # Wrap a callable in a coroutine
        @asyncio.coroutine
        def wrapper(*args, **kwargs):
            result = fn_or_coro(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return (yield from result)
            else:
                return result
        return wrapper
    else:
        raise ValueError('Parameter is not method, function or coroutine')

class Async(object):

    def __init__(self, work, *args, **kwargs):

        if isinstance(work, asyncio.Future):
            self._future = work
        elif asyncio.iscoroutine(work):
            self._future = asyncio.ensure_future(work)
        elif callable(work):
            self._future = asyncio.ensure_future(
                ensure_this_is_a_coroutine(work)(*args, **kwargs)
            )
        else:
            self._future = asyncio.ensure_future(
                ensure_this_is_a_coroutine(lambda: work)()
            )
        self._chained = None

    def bind(self, next_work: Any):
        next_work = ensure_this_is_a_coroutine(next_work)

        def resolved(func):
            try:
                res = check_result(func, self._chained)
            except TaskError:
                return
            t: asyncio.Future = asyncio.ensure_future(next_work(res))
            t.add_done_callback(lambda func: pass_result(func, new_future))

        new_future = asyncio.Future()
        next_async = Async(new_future)
        self._chained = new_future
        self._future.add_done_callback(resolved)
        return next_async

    def __rshift__(self, other):
        return self.bind(other)

    @property
    def future(self):
        return self._future