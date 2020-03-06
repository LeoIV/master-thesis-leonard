from asyncio import Future
from typing import Iterable


def check_finished_futures_and_return_unfinished(futures: Iterable[Future]):
    finished_futures = [f for f in futures if f.done()]
    unfinished_futures = [f for f in futures if f not in finished_futures]
    for f in finished_futures:
        f.result()
    return unfinished_futures
