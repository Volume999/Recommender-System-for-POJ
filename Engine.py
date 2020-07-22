import math
from pprint import pprint
import statistics
from collections import defaultdict
from functools import reduce
import pandas as pd
from enum import Enum
import CustomLib
import random
from Calculators import WeightCalculator, SimilarityCalculator
import sys
import time
import csv


def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:{} args:[{}, {}] took: {} sec'.format(f.__name__, args, kw, te-ts))
        return result

    return timed


class VerdictTypes(Enum):
    success = 1
    fail = 2


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


class ProblemTypes(Enum):
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


class Engine:
    class User:
        def __init__(self):
            self.problems_solved = list()
            self.problems_unsolved = list()
            self.submissions_stats = dict()
            self.user_type = UserTypes.variable
            self.projections = dict()
            self.similarities = list()
            self.recommendations = defaultdict(float)

    class SubmissionStats:
        def __init__(self, attempts):
            self.attempts = attempts
            self.submission_type = 0

    class ProblemStats:
        def __init__(self):
            self.attempts_before_fail = list()
            self.attempts_before_success = list()
            self.unsolved_threshold = 0
            self.solved_threshold = 0
            self.problem_type = ProblemTypes.variable

    class Variables:
        edge_weight_threshold = 0
        similarity_threshold = 0
        neighbourhood_size = 200
        recommendation_size = 10
        voting_strategy = VotingStrategy.simple
        similarity_strategy = SimilarityStrategy.adar_adamic
        path = str()

    class Data:
        def __init__(self):
            self.path = str()
            self.users = dict()
            self.problems = dict()

    class Testing:
        test_solve_requirement = 10

        def __init__(self, engine):
            self.engine = engine
            self.users_test = dict()
            self.f1_agg = list()
            self.recall_agg = list()
            self.p_agg = list()
            self.one_hit_agg = list()
            self.mrr_agg = list()

        def clear_aggregates(self):
            self.f1_agg = list()
            self.recall_agg = list()
            self.p_agg = list()
            self.one_hit_agg = list()
            self.mrr_agg = list()

        def clear_users_test(self):
            self.users_test = dict()

        @timeit
        def perform_test(self):
            precision = 0
            recall = 0
            count = 0
            one_hit = 0
            users = self.engine.data.users
            for (username, problemsSolved) in self.users_test.items():
                if len(users[username].recommendations.keys()) > 0:
                    count += 1
                    true_positive = 0
                    for probId in problemsSolved:
                        if probId in users[username].recommendations.keys():
                            true_positive += 1
                    precision += true_positive / len(users[username].recommendations.keys())
                    recall += true_positive / len(problemsSolved)
                    if true_positive > 0:
                        one_hit += 1
            mrr_list = list()
            for j in self.engine.testing.users_test:
                user = self.engine.data.users[j]
                for i in range(len(user.recommendations.keys())):
                    prob = list(user.recommendations.keys())[i]
                    if prob in self.users_test[j]:
                        mrr_list.append(1 / (i + 1))
                        break
            mrr = statistics.mean(mrr_list)
            if count != 0:
                precision = precision / count
                recall = recall / count
                f1 = 2 * precision * recall / (precision + recall)
                one_hit = one_hit / count
            else:
                precision = recall = f1 = 0
            self.p_agg.append(precision)
            self.recall_agg.append(recall)
            self.f1_agg.append(f1)
            self.one_hit_agg.append(one_hit)
            self.mrr_agg.append(mrr)

        def print_means(self):
            with open(file='Stats.csv', mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([self.test_solve_requirement,
                                self.engine.Variables.recommendation_size,
                                self.engine.Variables.similarity_strategy,
                                self.engine.Variables.voting_strategy,
                                self.p_agg, self.recall_agg,
                                self.f1_agg,
                                self.one_hit_agg,
                                self.mrr_agg])
            print("Precision: {}\nRecall: {}\nF1: {}\nOneHit: {}\nMRR: {}".format(
                statistics.mean(self.p_agg),
                statistics.mean(self.recall_agg),
                statistics.mean(self.f1_agg),
                statistics.mean(self.one_hit_agg),
                statistics.mean(self.mrr_agg)
            ))

    def __init__(self, path=""):
        self.data = self.Data()
        self.Variables.path = path
        self.data.path = path
        self.testing = self.Testing(self)

    def clear_data(self):
        self.data = self.Data()
        self.testing.clear_users_test()
        self.data.path = self.Variables.path

    @timeit
    def initialize(self):
        self.clear_data()
        df = pd.read_csv(self.data.path,
                         header=0,
                         names=['id', 'user', 'status', 'count'])
        for row in df.iterrows():
            prob_id = row[1][0]
            user = str(row[1][1])
            status = row[1][2]
            count = row[1][3]
            # print(tempProbId, tempUser, tempStatus)
            if count < 100:
                if user not in self.data.users.keys():
                    self.data.users[user] = self.User()
                if status == VerdictTypes.success.value:
                    self.data.users[user].problems_solved.append(prob_id)
                elif status == VerdictTypes.fail.value:
                    self.data.users[user].problems_unsolved.append(prob_id)
                self.data.users[user].submissions_stats[prob_id] = self.SubmissionStats(attempts=count)

                if prob_id not in self.data.problems:
                    self.data.problems[prob_id] = self.ProblemStats()
                if status == VerdictTypes.success.value:
                    self.data.problems[prob_id].attempts_before_success.append(count)
                elif status == VerdictTypes.fail.value:
                    self.data.problems[prob_id].attempts_before_fail.append(count)
        delete_users = list()
        for user in self.data.users:
            if len(self.data.users[user].problems_solved) < 5:
                # print(self.data.users[user].problems_solved)
                delete_users.append(user)
        # print(len(delete_users))
        for user in delete_users:
            self.data.users.pop(user)
        # for user in self.data.users:
        #     CustomLib.debug_print("User", user, self.data.users[user].problems_solved, self.data.users[user].problems_unsolved)

    # def initialize_tests(self):
    #     delete_users = list()
    #     for user in self.data.users:
    #         if len(self.data.users[user].problems_solved) < self.testing.test_solve_requirement * 2:
    #             delete_users.append(user)
    #     for user in delete_users:
    #         self.data.users.pop(user)
    #     for user in self.data.users:
    #         self.testing.users_test[user] = set()
    #         random.shuffle(self.data.users[user].problems_solved)
    #         self.data.users[user].problems_solved, self.testing.users_test[user] = CustomLib.split_in_half(
    #             self.data.users[user].problems_solved)
    #         for prob in self.testing.users_test[user]:
    #             self.data.users[user].submissions_stats.pop(prob)
    def initialize_tests(self):
        for user in self.data.users:
            if len(self.data.users[user].problems_solved) >= self.testing.test_solve_requirement * 2:
                self.testing.users_test[user] = set()
                random.shuffle(self.data.users[user].problems_solved)
                self.data.users[user].problems_solved, self.testing.users_test[user] = CustomLib.split_in_half(
                    self.data.users[user].problems_solved)
                for prob in self.testing.users_test[user]:
                    self.data.users[user].submissions_stats.pop(prob)

    @timeit
    def categorize_problems(self):
        for prob in self.data.problems:
            self.data.problems[prob].solved_threshold = 0 if len(self.data.problems[prob].attempts_before_success) == 0 \
                else statistics.mean(self.data.problems[prob].attempts_before_success)
            self.data.problems[prob].unsolved_threshold = 0 if len(self.data.problems[prob].attempts_before_fail) == 0 \
                else statistics.mean(self.data.problems[prob].attempts_before_fail)
            solved_with_many = len([val for val in self.data.problems[prob].attempts_before_success if
                                    val > self.data.problems[prob].solved_threshold])
            solved_with_little = len([val for val in self.data.problems[prob].attempts_before_success if
                                      val <= self.data.problems[prob].solved_threshold])
            if len(self.data.problems[prob].attempts_before_success) > 1:
                if solved_with_little >= 2 * solved_with_many:
                    self.data.problems[prob].problem_type = ProblemTypes.easy
                elif solved_with_many >= 2 * solved_with_little:
                    self.data.problems[prob].problem_type = ProblemTypes.difficult
                else:
                    self.data.problems[prob].problem_type = ProblemTypes.variable


    def categorize_user(self, user):
        for prob in user.problems_solved:
            if user.submissions_stats[prob].attempts >= self.data.problems[prob].solved_threshold:
                user.submissions_stats[prob].submission_type = SubmissionType.solved_with_many
            else:
                user.submissions_stats[prob].submission_type = SubmissionType.solved_with_few
        for prob in user.problems_unsolved:
            if user.submissions_stats[prob].attempts >= self.data.problems[prob].unsolved_threshold:
                user.submissions_stats[prob].submission_type = SubmissionType.unsolved_with_many
            else:
                user.submissions_stats[prob].submission_type = SubmissionType.unsolved_with_few
        solved_with_many = len([val for val in user.problems_solved if
                                user.submissions_stats[
                                    val].submission_type == SubmissionType.solved_with_many])
        # print([val for val in self.users[user].problems_solved])
        solved_with_few = len([val for val in user.problems_solved if
                               user.submissions_stats[
                                   val].submission_type == SubmissionType.solved_with_few])
        # CustomLib.debug_print("Submissions", user, solved_with_many, solved_with_few)
        if solved_with_many >= 2 * solved_with_few:
            user.user_type = UserTypes.imprecise
        elif solved_with_few >= 2 * solved_with_many:
            user.user_type = UserTypes.precise
        else:
            user.user_type = UserTypes.variable

    @timeit
    def categorize_users(self):
        for user in self.data.users:
            self.categorize_user(self.data.users[user])

    def manage_noise_user(self, user):
        for prob in user.problems_solved:
            if self.data.problems[prob].problem_type.value == user.user_type.value \
                    and self.data.problems[prob].problem_type != ProblemTypes.variable:
                user.submissions_stats[prob].submission_type = SubmissionType.solved_with_few \
                    if user.user_type == UserTypes.precise \
                    else SubmissionType.solved_with_many

    @timeit
    def manage_noise_users(self):
        for user in self.data.users:
            self.manage_noise_user(self.data.users[user])

    def get_user_projections(self, user):
        ans = dict()
        for j in self.data.users:
            user2 = self.data.users[j]
            if user is not user2:
                edge_weight = len([val for val in user.submissions_stats
                                   if val in user2.submissions_stats
                                   and user.submissions_stats[val].submission_type
                                   == user2.submissions_stats[val].submission_type])
                if edge_weight > self.Variables.edge_weight_threshold:
                    ans[user2] = edge_weight
                # CustomLib.debug_print("Projections", sorted(user.problems_solved), sorted(user2.problems_solved), sorted(user.problems_unsolved), sorted(user2.problems_unsolved), edge_weight)
        return ans

    @timeit
    def build_user_projections(self):
        for i in self.data.users.keys():
            self.data.users[i].projections = self.get_user_projections(self.data.users[i])

    def get_similarity_value(self, user1, user2):
        solutions = {
            SimilarityStrategy.jaccard_neighbours: SimilarityCalculator.calc_jaccard_neighbours,
            SimilarityStrategy.jaccard_problems: SimilarityCalculator.calc_jaccard_problems,
            SimilarityStrategy.preferential: SimilarityCalculator.calc_preferential,
            SimilarityStrategy.common_neighbours: SimilarityCalculator.calc_common_neighbours,
            SimilarityStrategy.edge_weight: SimilarityCalculator.calc_edge_weight,
            SimilarityStrategy.adar_adamic: SimilarityCalculator.calc_adar_atamic
        }
        if self.Variables.similarity_strategy not in solutions:
            raise Exception("Not a viable strategy")
        return solutions[self.Variables.similarity_strategy](user1, user2)

    # @timeit
    def get_user_similarities(self, user):
        ans = list()
        for j in self.data.users:
            user2 = self.data.users[j]
            if user is not user2:
                sim_value = self.get_similarity_value(user, user2)
                # CustomLib.debug_print("SimValue Check", sim_value)
                if not set(user2.problems_solved).issubset(user.problems_solved) \
                        and sim_value > self.Variables.similarity_threshold:
                    ans.append((user2, sim_value))
        ans.sort(key=lambda a: a[1], reverse=True)
        ans = CustomLib.first_k_elements(ans, self.Variables.neighbourhood_size)
        # CustomLib.debug_print("SimilaritiesCheck", ans)
        return ans

    @timeit
    def build_similarities(self):
        for i in self.data.users:
            # CustomLib.debug_print("Projections", i, self.data.users[i].projections)
            self.data.users[i].similarities = self.get_user_similarities(self.data.users[i])

    def get_weight_value(self, user, similar_user, sim_value):
        solutions = {
            VotingStrategy.simple: WeightCalculator.calc_simple_voting,
            VotingStrategy.weighted: WeightCalculator.calc_weighted_voting,
            VotingStrategy.positional: WeightCalculator.calc_positional_voting
        }
        if self.Variables.voting_strategy not in solutions:
            raise Exception("Not a viable voting strategy")
        return solutions[self.Variables.voting_strategy](user, similar_user, sim_value)

    def get_user_recommendations(self, user):
        ans = defaultdict(float)
        for (user2, simValue) in user.similarities:
            voting_weight = self.get_weight_value(user, user2, simValue)
            # CustomLib.debug_print("Vote check", voting_weight)
            for prob in user2.problems_solved:
                if prob not in user.problems_solved:
                    ans[prob] += voting_weight
        temp = list(ans.items())
        temp.sort(key=lambda a: a[1], reverse=True)
        temp = CustomLib.first_k_elements(temp, self.Variables.recommendation_size)
        ans = defaultdict(float)
        for (prob, count) in temp:
            ans[prob] = count

        return ans

    @timeit
    def build_recommendations(self):
        for i in self.data.users.keys():
            # CustomLib.debug_print("Similarities test", i, self.data.users[i].similarities)
            self.data.users[i].recommendations = self.get_user_recommendations(self.data.users[i])
            # CustomLib.debug_print("Recommendations check", i, self.data.users[i].recommendations)

    def execute_for_user(self, user_data):
        user = self.User()
        for (pid, status, count) in user_data:
            if pid in self.data.problems:
                (user.problems_solved if status == 1 else user.problems_unsolved).append(pid)
                user.submissions_stats[pid] = self.SubmissionStats(attempts=count)
        self.categorize_user(user)
        self.manage_noise_user(user)
        user.projections = self.get_user_projections(user)
        user.similarities = self.get_user_similarities(user)
        user.recommendations = self.get_user_recommendations(user)
        print(user.recommendations)
        # CustomLib.debug_print("Executing for user",
        #                       user_data,
        #                       user.problems_solved,
        #                       user.problems_unsolved,
        #                       user.submissions_stats,
        #                       user.projections,
        #                       user.similarities,
        #                       user.recommendations)
                              # self.data.users)

    def execute(self):
        self.categorize_problems()
        self.categorize_users()
        self.manage_noise_users()
        self.build_user_projections()
        self.build_similarities()
        self.build_recommendations()

    def run(self):
        self.initialize()
        self.execute()

    def test(self):
        full_test = False
        if full_test:
            for similarity in SimilarityStrategy:
                for voting in VotingStrategy:
                    self.Variables.similarity_strategy = similarity
                    self.Variables.voting_strategy = voting
                    print("Similarity = {}, Voting = {}".format(self.Variables.similarity_strategy,
                                                                self.Variables.voting_strategy))
                    self.testing.clear_aggregates()
                    for i in range(3):
                        self.initialize()
                        self.initialize_tests()
                        self.execute()
                        self.testing.perform_test()
                    print("Users size:", len(self.data.users))
                    print("Test size:", len(self.testing.users_test))
                    self.testing.print_means()
        else:
            self.Variables.similarity_strategy = SimilarityStrategy.jaccard_neighbours
            self.Variables.voting_strategy = VotingStrategy.weighted
            print("Similarity = {}, Voting = {}".format(self.Variables.similarity_strategy,
                                                        self.Variables.voting_strategy))
            self.testing.clear_aggregates()
            for i in range(1):
                self.initialize()
                self.initialize_tests()
                self.execute()
                self.testing.perform_test()
                users = self.data.users
                # for username in self.data.users:
                #     CustomLib.debug_print("Full test", username,
                #                           sorted(users[username].problems_solved),
                #                           sorted(users[username].recommendations.keys()),
                #                           sorted(self.testing.users_test[username]))

            print("Users size:", len(self.data.users))
            print("Test size:", len(self.testing.users_test))
            self.testing.print_means()
