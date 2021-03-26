from Enums import SubmissionType, VerdictTypes
import pandas as pd
from Variables import Variables
from Structs import User, SubmissionStats, ProblemStats
from DataSource import DataSourceCsv
import pickle
from Enums import ImportMode


# Data collection logic
class Collection:
    def __init__(self):
        self.users = dict()
        self.problems = dict()

    def import_from_csv(self, path):
        users = self.users
        problems = self.problems
        df = pd.read_csv(path,
                         header=0,
                         names=['id', 'user', 'status', 'count'])
        for row in df.iterrows():  # O(rows)
            prob_id = row[1][0]
            user = str(row[1][1])
            status = row[1][2]
            count = row[1][3]
            # if the number of attempts for a problem is more than 100
            # it is not considered
            if user not in users.keys():  # O(1)
                users[user] = User()
            if status == VerdictTypes.success.value:
                users[user].problems_solved.append(prob_id)
            elif status == VerdictTypes.fail.value or status == VerdictTypes.partially_solved.value:
                users[user].problems_unsolved.append(prob_id)
            users[user].submissions_stats[prob_id] = SubmissionStats(attempts=count)
            if status == VerdictTypes.partially_solved.value:
                users[user].submissions_stats[prob_id].submission_type = SubmissionType.solved_partially

            if prob_id not in problems:
                problems[prob_id] = ProblemStats()
            if status == VerdictTypes.success.value:
                problems[prob_id].attempts_before_success.append(count)
            elif status == VerdictTypes.fail.value or status == VerdictTypes.partially_solved.value:
                problems[prob_id].attempts_before_fail.append(count)
        self.users = dict(
            filter(lambda user: len(user[1].problems_solved) >= Variables.user_solve_requirements, self.users.items()))


def get_Collection(data_source):
    if isinstance(data_source, DataSourceCsv):
        result = Collection()
        result.import_from_csv(data_source.file_path)
        return result
    else:
        return None
