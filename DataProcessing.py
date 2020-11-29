import statistics
from Enums import ProblemDifficulty, SubmissionType, UserTypes


# Preprocessing data for more accurate results
class Preprocessing:
    def __init__(self, data):
        self.problems = data.problems
        self.users = data.users

    # Categorizing problems into Easy, Difficult and variable
    def categorize_problems(self):
        problems = self.problems
        for prob in problems:
            problems[prob].solved_threshold = 0 \
                if len(problems[prob].attempts_before_success) == 0 \
                else statistics.mean(problems[prob].attempts_before_success)
            problems[prob].unsolved_threshold = 0 if len(problems[prob].attempts_before_fail) == 0 \
                else statistics.mean(problems[prob].attempts_before_fail)
            # Solved with many, with little - number of times a problem was solved with attempts
            # higher/lower that the mean
            solved_with_many = len([val for val in problems[prob].attempts_before_success if
                                    val > problems[prob].solved_threshold])
            solved_with_little = len([val for val in problems[prob].attempts_before_success if
                                      val <= problems[prob].solved_threshold])
            # If the task is solved with less attempts than the mean twice as
            # frequent as solved with more attempts than mean, then its easy
            # If the task is solved with more attempts than mean twice as
            # much. then its difficult
            # Else its variable
            if len(problems[prob].attempts_before_success) > 1:
                if solved_with_little >= 2 * solved_with_many:
                    problems[prob].problem_type = ProblemDifficulty.easy
                elif solved_with_many >= 2 * solved_with_little:
                    problems[prob].problem_type = ProblemDifficulty.difficult
                else:
                    problems[prob].problem_type = ProblemDifficulty.variable

    # Categorizing users into Precise, Imprecise and Variable
    def categorize_user(self, user):
        problems = self.problems
        for prob in user.problems_solved:
            if user.submissions_stats[prob].attempts >= problems[prob].solved_threshold:
                user.submissions_stats[prob].submission_type = SubmissionType.solved_with_many
            else:
                user.submissions_stats[prob].submission_type = SubmissionType.solved_with_few
        for prob in user.problems_unsolved:
            if user.submissions_stats[prob].submission_type != SubmissionType.solved_partially:
                if user.submissions_stats[prob].attempts >= problems[prob].unsolved_threshold:
                    user.submissions_stats[prob].submission_type = SubmissionType.unsolved_with_many
                else:
                    user.submissions_stats[prob].submission_type = SubmissionType.unsolved_with_few
        solved_with_many = len([val for val in user.problems_solved if
                                user.submissions_stats[
                                    val].submission_type == SubmissionType.solved_with_many])
        solved_with_few = len([val for val in user.problems_solved if
                               user.submissions_stats[
                                   val].submission_type == SubmissionType.solved_with_few])
        # If the used solved twice as many tasks with attempts less than its
        # that problem's mean, then the user is precise
        # If the used solved twice as many tasks with attempts more than its
        # that problem's mean, then the user is imprecise
        if solved_with_many >= 2 * solved_with_few:
            user.user_type = UserTypes.imprecise
        elif solved_with_few >= 2 * solved_with_many:
            user.user_type = UserTypes.precise
        else:
            user.user_type = UserTypes.variable

    def categorize_users(self):
        for user in self.users:
            self.categorize_user(self.users[user])

    # For each submission
    # if the user is precise, and the problem is easy
    # then the submission must have solved_with_few status
    # if the user is imprecise, and the problem is hard
    # then the submission must have solved_with_many status
    # and vice-versa
    def manage_noise_user(self, user):
        problems = self.problems
        for prob in user.problems_solved:
            if problems[prob].problem_type.value == user.user_type.value \
                    and problems[prob].problem_type != ProblemDifficulty.variable:
                user.submissions_stats[prob].submission_type = \
                    SubmissionType.solved_with_few \
                    if user.user_type == UserTypes.precise \
                    else SubmissionType.solved_with_many

    def manage_noise_users(self):
        for user in self.users:
            self.manage_noise_user(self.users[user])

    # The Preprocessing is done in three steps
    # 1. categorizing problems
    # 2. categorizing users
    # 3. noise detection
    def preprocess(self):
        self.categorize_problems()
        self.categorize_users()
        self.manage_noise_users()

    def preprocess_user(self, user):
        self.categorize_user(user)
        self.manage_noise_user(user)
