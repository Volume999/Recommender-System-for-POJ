from Enums import SubmissionType, VerdictTypes
import pandas as pd
from Variables import Variables
from Structs import User, SubmissionStats, ProblemStats
from Enums import ImportMode


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
        for row in df.iterrows():
            prob_id = row[1][0]
            user = str(row[1][1])
            status = row[1][2]
            count = row[1][3]
            if count < 100:
                if user not in users.keys():
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
        delete_users = list()
        for user in users:
            if len(users[user].problems_solved) < Variables.user_solve_requirements:
                delete_users.append(user)
        for user in delete_users:
            users.pop(user)

