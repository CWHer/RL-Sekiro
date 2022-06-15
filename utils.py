import time
from functools import wraps


def timeLog(func):
    @wraps(func)
    def clocked(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print("{} finishes after {:.2f} s".format(
            func.__name__, time.time() - start))
        return ret
    return clocked
