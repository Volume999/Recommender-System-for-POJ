import csv
import random
import CustomLib
from Variables import Variables
import statistics


class Testing:
    test_solve_requirement = 5

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
        users = self.engine.data.users
        tests = self.users_test
        precision = 0
        recall = 0
        one_hit = 0
        for (username, problemsSolved) in tests.items():
            user_recommendations = [prob[0] for prob in users[username].recommendations]
            if len(user_recommendations) > 0:
                true_positive = 0
                for probId in problemsSolved:
                    if probId in user_recommendations:
                        true_positive += 1
                precision += true_positive / len(user_recommendations)
                recall += true_positive / len(problemsSolved)
                if true_positive > 0:
                    one_hit += 1
        mrr_list = list()
        for j in tests:
            user = users[j]
            user_r = [prob[0] for prob in user.recommendations]
            for i in range(len(user_r)):
                prob = list(user_r)[i]
                if prob in tests[j]:
                    mrr_list.append(1 / (i + 1))
                    break
        mrr = statistics.mean(mrr_list) if len(mrr_list) > 0 else 0
        precision /= len(tests)
        recall /= len(tests)
        one_hit /= len(tests)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        self.p_agg.append(precision)
        self.recall_agg.append(recall)
        self.f1_agg.append(f1)
        self.one_hit_agg.append(one_hit)
        self.mrr_agg.append(mrr)

    def print_means(self):
        with open(file='stats.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([self.test_solve_requirement,
                             Variables.recommendation_size,
                             Variables.similarity_strategy,
                             Variables.voting_strategy,
                             Variables.edge_weight_threshold,
                             self.p_agg,
                             self.recall_agg,
                             self.f1_agg,
                             self.one_hit_agg,
                             self.mrr_agg])

    def initialize_tests(self):
        self.users_test = dict()
        users = self.engine.data.users
        for user in users:
            if len(users[user].problems_solved) >= self.test_solve_requirement * 2:
                self.users_test[user] = set()
                random.Random(228).shuffle(users[user].problems_solved)
                users[user].problems_solved, self.users_test[user] = CustomLib.split_in_half(
                    users[user].problems_solved)
                for prob in self.users_test[user]:
                    users[user].submissions_stats.pop(prob)
