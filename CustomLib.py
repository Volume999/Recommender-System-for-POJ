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