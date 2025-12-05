# ================================================================
# parallel.py
# Simple multiprocessing for GPU chunk batching
# ================================================================
import multiprocessing as mp

def parallel_map(func, items, workers=4):
    """
    Apply func(item) across items in parallel CPU workers.
    GPU alignment is launched within the worker process.
    """
    with mp.Pool(workers) as pool:
        for result in pool.imap(func, items):
            yield result
