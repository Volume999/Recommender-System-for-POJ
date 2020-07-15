import math
from pprint import pprint
import statistics
from collections import defaultdict
from functools import reduce
import pandas as pd
from enum import Enum
import CustomLib
import random


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

    class WeightCalculator:
        def __init__(self, engine):
            self.engine = engine

        @staticmethod
        def calc_simple_voting(user, similar_user, sim_value):
            return 1

        def calc_weighted_voting(self, user, similar_user, sim_value):
            return float(sim_value) / reduce(lambda acc, item: acc + item[1], self.engine.data.similarity[user], 0)

        def calc_positional_voting(self, user, similar_user, sim_value):
            # CustomLib.debug_print("Testing voting", user, similar_user,
            # self.engine.similarity[user], 1.0 /
            # (self.engine.similarity[user].index((similar_user, sim_value)) + 1))
            # exit(0)
            return 1.0 / (self.engine.data.similarity[user].index((similar_user, sim_value)) + 1)

    class SimilarityCalculator:
        def __init__(self, engine):
            self.engine = engine

        def calc_jaccard_neighbours(self, user1, user2):
            user1_neighbours = self.engine.data.users_projection_matrix[user1].keys()
            user2_neighbours = self.engine.data.users_projection_matrix[user2].keys()
            intersection_val = CustomLib.intersection_length(user1_neighbours, user2_neighbours)
            union_val = CustomLib.union_length(user1_neighbours, user2_neighbours)
            ans = 0 if union_val == 0 else intersection_val / union_val
            # if ans != 0.9354838709677419:
            #     CustomLib.debug_print("Jaccard Neighbours Test",
            #     user1, self.users[user1],
            #     users_projection_matrix[user1],
            #     user2, self.users[user2],
            #     users_projection_matrix[user2],
            #     intersection_val,
            #     union_val,
            #     ans)
            return ans

        def calc_jaccard_problems(self, user1, user2):
            if user2 not in self.engine.data.users_projection_matrix[user1]:
                return 0
            intersection_val = self.engine.data.users_projection_matrix[user1][user2]
            union_val = CustomLib.union_length(self.engine.data.users[user1].problems_solved + self.engine.data.users[user1].problems_unsolved,
                                               self.engine.data.users[user2].problems_solved + self.engine.data.users[user2].problems_unsolved)
            ans = 0 if union_val == 0 else intersection_val / union_val
            # CustomLib.debug_print("Jaccard Problems Test", user1, user2, ans)
            return ans

        def calc_edge_weight(self, user1, user2):
            return 0 if user2 not in self.engine.data.users_projection_matrix[user1].keys() \
                else self.engine.data.users_projection_matrix[user1][user2]

        def calc_common_neighbours(self, user1, user2):
            intersection = CustomLib.intersection(self.engine.data.users_projection_matrix[user1],
                                                  self.engine.data.users_projection_matrix[user2])
            ans = reduce(lambda acc, user: acc + self.engine.data.users_projection_matrix[user1][user] +
                                           self.engine.data.users_projection_matrix[user2][user], intersection, 0)
            return ans

        def calc_adar_atamic(self, user1, user2):
            user1_neighbourhood = self.engine.data.users_projection_matrix[user1]
            user2_neighbourhood = self.engine.data.users_projection_matrix[user2]
            intersection = CustomLib.intersection(user1_neighbourhood,
                                                  user2_neighbourhood)
            ans = reduce(lambda acc, user: acc + (user1_neighbourhood[user] + user2_neighbourhood[user])
                                           / math.log(1 + reduce(lambda acc2, i: acc2 + i,
                                                                 self.engine.data.users_projection_matrix[
                                                                     user].values(),
                                                                 0)
                                                      ),
                         intersection,
                         0)
            # CustomLib.debug_print("Adar Atamic Testing", user1, users[user1],
            # users_projection_matrix[user1], user2, users[user2],
            # users_projection_matrix[user2], intersection, ans)
            return ans

        def calc_preferential(self, user1, user2):
            ans = reduce(lambda acc, i: acc + i, self.engine.data.users_projection_matrix[user1].values(), 0) * \
                  reduce(lambda acc, i: acc + i, self.engine.data.users_projection_matrix[user2].values(), 0)
            # CustomLib.debug_print("Preferential Strategy Test", user1,
            # users[user1], users_projection_matrix[user1], user2, users[user2],
            # users_projection_matrix[user2], ans)
            return ans

    class Variables:
        edge_weight_threshold = 3
        similarity_threshold = 0
        neighbourhood_size = 50
        recommendation_size = 10
        voting_strategy = VotingStrategy.simple
        similarity_strategy = SimilarityStrategy.preferential
        path = str()

    class Data:
        def __init__(self):
            self.path = str()
            self.users = dict()
            self.users_projection_matrix = dict()
            self.similarity = dict()
            self.recommendations = dict()
            self.problems = dict()

    class Testing:
        test_solve_requirement = 3

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

        def perform_test(self):
            precision = 0
            recall = 0
            count = 0
            one_hit = 0
            for (username, problemsSolved) in self.users_test.items():
                if username in self.engine.data.recommendations.keys() and len(
                        self.engine.data.recommendations[username].keys()) > 0:
                    # CustomLib.debug_print("Test Candidate", username,
                    # self.users[username].problems_solved,
                    # self.recommendations[username], problemsSolved)
                    # if CustomLib.intersection(recommendations[username].keys(),
                    # problemsSolved) != len(problemsSolved):
                    #     CustomLib.debug_print("Testing Solution",
                    #     sorted(recommendations[username].keys()),
                    #     sorted(problemsSolved))
                    count += 1
                    true_positive = 0
                    # print(recommendations[username].keys())
                    # print("{} : {} - {}".format(username, recommendations[username], problemsSolved))
                    for probId in problemsSolved:
                        if probId in self.engine.data.recommendations[username].keys():
                            # print(username, probId)
                            true_positive += 1
                    precision += true_positive / len(self.engine.data.recommendations[username].keys())
                    recall += true_positive / len(problemsSolved)
                    if true_positive > 0:
                        one_hit += 1
                    # else:
                    #     CustomLib.debug_print("one_hit",
                    #     username, self.users[username],
                    #     self.users_projection_matrix[username],
                    #     self.similarity[username],
                    #     self.recommendations[username])
                    #     for a in self.users_projection_matrix[username].keys():
                    #         CustomLib.debug_print("candidate", a, self.users[a])
            # print(precision, count)
            mrr_list = list()
            for user in self.engine.data.recommendations:
                for i in range(len(self.engine.data.recommendations[user].keys())):
                    prob = list(self.engine.data.recommendations[user].keys())[i]
                    if prob in self.users_test[user]:
                        mrr_list.append(1 / (i + 1))
                        break
                # if mrr != 1:
                #     CustomLib.debug_print("Checking mrr", user, self.recommendations[user], self.users_test[user], mrr)
                # exit(0)
            mrr = statistics.mean(mrr_list)
            if count != 0:
                precision = precision / count
                recall = recall / count
                f1 = 2 * precision * recall / (precision + recall)
                one_hit = one_hit / count
            else:
                precision = recall = f1 = 0
            # CustomLib.debug_print("asd", precision, recall, f1, one_hit, count)
            # exit()
            self.p_agg.append(precision)
            self.recall_agg.append(recall)
            self.f1_agg.append(f1)
            self.one_hit_agg.append(one_hit)
            self.mrr_agg.append(mrr)
            # print("Precision = {}, Recall = {}, F1 = {}, One Hit = {}, Count = {}".format(precision, recall, f1, one_hit,
            #                                                                               count))
            # for i in recommendations:
            #     print(i, len(recommendations[i]), recommendations[i])

        def print_means(self):
            CustomLib.debug_print("testing", len(self.p_agg), len(self.recall_agg), len(self.f1_agg), len(self.one_hit_agg), len(self.mrr_agg))
            # exit()
            print("Precision: {}\nRecall: {}\nF1: {}\nOneHit: {}\nMRR: {}".format(
                statistics.mean(self.p_agg),
                statistics.mean(self.recall_agg),
                statistics.mean(self.f1_agg),
                statistics.mean(self.one_hit_agg),
                statistics.mean(self.mrr_agg)
            ))

    def __init__(self, path=""):
        self.weight_calculator = self.WeightCalculator(self)
        self.similarity_calculator = self.SimilarityCalculator(self)
        self.data = self.Data()
        self.Variables.path = path
        self.data.path = path
        self.testing = self.Testing(self)

    def clear_data(self):
        self.data = self.Data()
        self.testing.clear_users_test()
        self.data.path = self.Variables.path

    def initialize(self):
        self.clear_data()
        # CustomLib.debug_print("initialize test", path, test_size, solve_requirement)
        df = pd.read_csv(self.data.path,
                         header=0,
                         names=['id', 'user', 'status', 'count'])
        # print(df)
        # exit(0)
        for row in df.iterrows():
            prob_id = row[1][0]
            user = row[1][1]
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

    def initialize_tests(self):
        delete_users = list()
        for user in self.data.users:
            if len(self.data.users[user].problems_solved) < self.testing.test_solve_requirement * 2:
                delete_users.append(user)
        for user in delete_users:
            self.data.users.pop(user)
        for user in self.data.users:
            self.testing.users_test[user] = set()
            # CustomLib.debug_print("before splitting",
            # self.users[user].problems_solved,
            # self.users[user].problems_unsolved,
            # self.users[user].submissions_stats.keys())
            random.shuffle(self.data.users[user].problems_solved)
            self.data.users[user].problems_solved, self.testing.users_test[user] = CustomLib.split_in_half(
                self.data.users[user].problems_solved)
            for prob in self.testing.users_test[user]:
                self.data.users[user].submissions_stats.pop(prob)
            # CustomLib.debug_print("splitting test",
            # self.users[user].problems_solved,
            # self.users_test[user], self.users[user].submissions_stats.keys())

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

    def categorize_users(self):
        for user in self.data.users:
            for prob in self.data.users[user].problems_solved:
                if self.data.users[user].submissions_stats[prob].attempts >= self.data.problems[prob].solved_threshold:
                    self.data.users[user].submissions_stats[prob].submission_type = SubmissionType.solved_with_many
                else:
                    self.data.users[user].submissions_stats[prob].submission_type = SubmissionType.solved_with_few
            for prob in self.data.users[user].problems_unsolved:
                if self.data.users[user].submissions_stats[prob].attempts >= self.data.problems[
                    prob].unsolved_threshold:
                    self.data.users[user].submissions_stats[prob].submission_type = SubmissionType.unsolved_with_many
                else:
                    self.data.users[user].submissions_stats[prob].submission_type = SubmissionType.unsolved_with_few
            solved_with_many = len([val for val in self.data.users[user].problems_solved if
                                    self.data.users[user].submissions_stats[
                                        val].submission_type == SubmissionType.solved_with_many])
            # print([val for val in self.users[user].problems_solved])
            solved_with_few = len([val for val in self.data.users[user].problems_solved if
                                   self.data.users[user].submissions_stats[
                                       val].submission_type == SubmissionType.solved_with_few])
            # CustomLib.debug_print("Submissions", user, solved_with_many, solved_with_few)
            if solved_with_many >= 2 * solved_with_few:
                self.data.users[user].user_type = UserTypes.imprecise
            elif solved_with_few >= 2 * solved_with_many:
                self.data.users[user].user_type = UserTypes.precise
            else:
                self.data.users[user].user_type = UserTypes.variable

    def manage_noise(self):
        for user in self.data.users:
            for prob in self.data.users[user].problems_solved:
                if self.data.problems[prob].problem_type.value == self.data.users[user].user_type.value \
                        and self.data.problems[prob].problem_type != ProblemTypes.variable:
                    self.data.users[user].submissions_stats[prob].submission_type = SubmissionType.solved_with_few \
                        if self.data.users[user].user_type == UserTypes.precise \
                        else SubmissionType.solved_with_many

    def build_user_projection_matrix(self):
        # for prob in self.problems:
        #     CustomLib.debug_print("Problems check",
        #     prob, self.problems[prob].attempts_before_success,
        #     self.problems[prob].attempts_before_fail,
        #     self.problems[prob].solved_threshold,
        #     self.problems[prob].unsolved_threshold,
        #     self.problems[prob].problem_type)
        # for user in self.users:
        #     CustomLib.debug_print("Users check",
        #     user, self.users[user].problems_solved,
        #     self.users[user].problems_unsolved)
        #     for p in self.users[user].problems_solved:
        #         CustomLib.debug_print("Users problems check",
        #         p,
        #         self.users[user].submissions_stats[p].attempts,
        #         self.problems[p].solved_threshold,
        #         self.users[user].submissions_stats[p].submission_type,
        #         self.users[user].user_type, self.problems[p].problem_type)
        # print(self.users.keys())
        for i in self.data.users.keys():
            if i not in self.data.users_projection_matrix:
                self.data.users_projection_matrix[i] = dict()
            for j in self.data.users:
                if i != j:
                    # CustomLib.debug_print("user projection matrix",
                    # self.users[i].problems_solved, self.users[j].problems_solved)
                    # edge_weight = CustomLib.intersection_length(self.users[i].problems_solved,
                    # self.users[j].problems_solved)
                    edge_weight = len([val for val in self.data.users[i].submissions_stats
                                       if val in self.data.users[j].submissions_stats
                                       and self.data.users[i].submissions_stats[val].submission_type
                                       == self.data.users[j].submissions_stats[val].submission_type])
                    if edge_weight >= self.Variables.edge_weight_threshold:
                        self.data.users_projection_matrix[i][j] = edge_weight
        # pprint.pprint(self.users_projection_matrix)
        # exit(0)

    def get_similarity_value(self, user1, user2):
        solutions = {
            SimilarityStrategy.jaccard_neighbours: self.similarity_calculator.calc_jaccard_neighbours,
            SimilarityStrategy.jaccard_problems: self.similarity_calculator.calc_jaccard_problems,
            SimilarityStrategy.preferential: self.similarity_calculator.calc_preferential,
            SimilarityStrategy.common_neighbours: self.similarity_calculator.calc_common_neighbours,
            SimilarityStrategy.edge_weight: self.similarity_calculator.calc_edge_weight,
            SimilarityStrategy.adar_adamic: self.similarity_calculator.calc_adar_atamic
        }
        if self.Variables.similarity_strategy not in solutions:
            raise Exception("Not a viable strategy")
        return solutions[self.Variables.similarity_strategy](user1, user2)

    def build_similarity_matrix(self):
        for i in self.data.users:
            if i not in self.data.similarity.keys():
                self.data.similarity[i] = list()
            for j in self.data.users:
                if i != j:
                    sim_value = self.get_similarity_value(i, j)
                    # print("Entry: {} {} {} {} {}".format(self.data.users[i].problems_solved,
                    # self.data.users[j].problems_solved, i, j, sim_value))
                    if not set(self.data.users[j].problems_solved).issubset(self.data.users[i].problems_solved) \
                            and sim_value > self.Variables.similarity_threshold:
                        self.data.similarity[i].append((j, sim_value))
                    # else:
                    # print("Subset: {} {} {}".format(self.users[i], self.users[j], sim_value))
        for i in self.data.similarity:
            self.data.similarity[i].sort(key=lambda a: a[1], reverse=True)
            self.data.similarity[i] = CustomLib.first_k_elements(self.data.similarity[i],
                                                                 self.Variables.neighbourhood_size)
            # [similarity[i][j] for j in range(len(similarity[i])) if j < TOP_K]

    def get_weight_value(self, user, similar_user, sim_value):
        solutions = {
            VotingStrategy.simple: self.weight_calculator.calc_simple_voting,
            VotingStrategy.weighted: self.weight_calculator.calc_weighted_voting,
            VotingStrategy.positional: self.weight_calculator.calc_positional_voting
        }
        if self.Variables.voting_strategy not in solutions:
            raise Exception("Not a viable voting strategy")
        return solutions[self.Variables.voting_strategy](user, similar_user, sim_value)

    def build_recommendation_matrix(self):
        for i in self.data.users.keys():
            if i not in self.data.recommendations.keys():
                self.data.recommendations[i] = defaultdict(float)
            for (userID, simValue) in self.data.similarity[i]:
                voting_weight = self.get_weight_value(i, userID, simValue)
                for prob in self.data.users[userID].problems_solved:
                    if prob not in self.data.users[i].problems_solved:
                        self.data.recommendations[i][prob] += voting_weight
            #     CustomLib.debug_print("data.recommendations",
            #                           i,
            #                           userID,
            #                           self.data.users[i].problems_solved,
            #                           self.data.users[i].problems_unsolved,
            #                           self.data.users[userID].problems_solved,
            #                           self.data.users[userID].problems_unsolved,
            #                           simValue,
            #                           self.get_weight_value(i, userID, simValue)
            #                           )
            # exit(0)
        # for i in self.users:
        #     CustomLib.debug_print("data.recommendations values test", i, self.similarity[i], recommendations[i])
        # for (j, sim_value) in self.similarity[i]:
        #     CustomLib.debug_print("Values in self.similarity", j, sorted(self.users[j]), sim_value)
        # exit(0)
        for i in self.data.recommendations:
            temp = list(self.data.recommendations[i].items())
            temp.sort(key=lambda a: a[1], reverse=True)
            temp = CustomLib.first_k_elements(temp, self.Variables.recommendation_size)
            # print("{}: {}".format(i, temp))
            self.data.recommendations[i] = dict()
            for (prob, count) in temp:
                self.data.recommendations[i][prob] = count
            # CustomLib.debug_print("Recommendations", i, sorted(list(self.recommendations[i].keys())))
            # exit(0)

    def execute(self):
        self.categorize_problems()
        self.categorize_users()
        self.manage_noise()
        self.build_user_projection_matrix()
        self.build_similarity_matrix()
        self.build_recommendation_matrix()

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
                    for i in range(30):
                        self.initialize()
                        self.initialize_tests()
                        self.execute()
                        self.testing.perform_test()
                    self.testing.print_means()
        else:
            self.Variables.similarity_strategy = SimilarityStrategy.jaccard_problems
            self.Variables.voting_strategy = VotingStrategy.simple
            print("Similarity = {}, Voting = {}".format(self.Variables.similarity_strategy,
                                                        self.Variables.voting_strategy))
            self.testing.clear_aggregates()
            for i in range(30):
                self.initialize()
                self.initialize_tests()
                self.execute()
                self.testing.perform_test()
            self.testing.print_means()
