from enum import Enum


class ImportMode(Enum):
    from_csv: 1
    saved_data: 2
    from_database: 3


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