from Enums import UserTypes, ProblemDifficulty


class User:
    def __init__(self):
        self.problems_solved = list()
        self.problems_unsolved = list()
        self.submissions_stats = dict()
        self.user_type = UserTypes.variable
        self.projections = dict()
        self.similarities = list()
        self.recommendations = list()


class SubmissionStats:
    def __init__(self, attempts):
        self.attempts = attempts
        self.submission_type = 0


class ProblemStats:
    def __init__(self):
        self.attempts_before_fail = list()
        self.attempts_before_success = list()
        # Unsolved, solved thresholds - means of attempts taken
        # or means of attempts before first solved submission
        self.unsolved_threshold = 0
        self.solved_threshold = 0
        # Problem type - Easy / Difficult / Variable
        self.problem_type = ProblemDifficulty.variable
        # Problem projections - SEE IF THIS IS USED ANYWHERE
        self.projections = list()
