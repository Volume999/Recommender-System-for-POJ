from enum import Enum


# Import modes:
# 1 - submission data from database in csv file to initialize engine
# 2 - load a ready to use model from blob
# 3 - load submission data from db directly
class ImportMode(Enum):
    from_csv: 1
    saved_data: 2
    from_database: 3


# Submission Status
class VerdictTypes(Enum):
    success = 1
    fail = 2
    partially_solved = 3


# Similarity strategy:
# Calculates the similarity between the users
# based on their submission statuses
class SimilarityStrategy(Enum):
    edge_weight = 1
    common_neighbours = 2
    jaccard_neighbours = 3
    jaccard_problems = 4
    adar_adamic = 5
    preferential = 6


# Voting strategy:
# Calculates the recommendation list based on the
# neighbours and their similarity values
class VotingStrategy(Enum):
    simple = 1
    weighted = 2
    positional = 3


# Problem difficulty:
# easy - when users generally solve the problem with few attempts
# difficult - when users generally solve the problem with many attempts
# variable - neither easy nor difficult
class ProblemDifficulty(Enum):
    easy = 1
    difficult = 2
    variable = -1


# User type assigns a type to user
# precise is that whose submissions are solved with few attempts generally
# imprecise is that whose submissions are solved with many attempts
# variable is that who is neither precise nor imprecise
class UserTypes(Enum):
    precise = 1
    imprecise = 2
    variable = -1


# Submission type calculates gives the type to the submission
# many or few is decided by whether the submission is correct
# with less than average attempts or more
class SubmissionType(Enum):
    solved_with_few = 1
    solved_with_many = 2
    unsolved_with_many = 3
    unsolved_with_few = 4
    solved_partially = 5
