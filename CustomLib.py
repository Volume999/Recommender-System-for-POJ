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


class VerdictTypes(Enum):
    success = 1
    fail = 2
    partially_solved = 3


class SimilarityStrategy(Enum):
    edge_weight = 1
    common_neighbours = 2
    jaccard_neighbours = 3
    jaccard_problems = 4
    adar_adamic = 5
    preferential = 6


class VotingStrategy(Enum):
    simple = 1
    weighted = 2
    positional = 3


class ProblemDifficulty(Enum):
    easy = 1
    difficult = 2
    variable = -1


class UserTypes(Enum):
    precise = 1
    imprecise = 2
    variable = -1


class SubmissionType(Enum):
    solved_with_few = 1
    solved_with_many = 2
    unsolved_with_many = 3
    unsolved_with_few = 4
    solved_partially = 5


# class ProblemType(Enum):
#     easy = 1
#     medium = 2
