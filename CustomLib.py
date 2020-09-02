from enum import Enum


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


def first_k_elements(lst, k):
    return [lst[j] for j in range(len(lst)) if j < k]


def split_in_half(lst):
    half = len(lst) // 2
    return lst[:half], lst[half:]

# def timeit(f):
#     def timed(*args, **kw):
#         ts = time.time()
#         result = f(*args, **kw)
#         te = time.time()
#
#         print('func:{} args:[{}, {}] took: {} sec'.format(f.__name__, args, kw, te - ts))
#         return result
#
#     return timed


# class ProblemType(Enum):
#     easy = 1
#     medium = 2
