from asyncio import Future
from typing import Iterable


def check_finished_futures_and_return_unfinished(futures: Iterable[Future]):
    """
    Call result on finished futures to get the error if thrown
    :param futures: the futures
    :return: list of unfinished futures
    """
    finished_futures = [f for f in futures if f.done()]
    unfinished_futures = [f for f in futures if not f.done()]
    for f in finished_futures:
        f.result()
    return unfinished_futures
