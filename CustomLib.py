from enum import Enum
from timeit import default_timer as timer
from datetime import timedelta


def intersection_length(lst1, lst2):
    return len([val for val in lst1 if val in lst2])


def union_length(lst1, lst2):
    return len(set(list(lst1) + list(lst2)))


def intersection(lst1, lst2):
    return [val for val in lst1 if val in lst2]


def debug_print(*args):
    print(args[0])
    for arg in range(1, len(args)):
        print("arg {}: {}".format(arg, args[arg]))


def split_in_half(lst):
    half = len(lst) // 2
    return lst[:half], lst[half:]


# timing the function
def debug_timing(func):
    def new_func(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        print(f"Time for function {func.__name__}: {timedelta(seconds=end - start)}")
        return result
    return new_func
