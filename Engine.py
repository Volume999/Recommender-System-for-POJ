import math
import statistics
from collections import defaultdict
from functools import reduce
import pandas as pd

import CustomLib


class SimilarityStrategy:
    edge_weight = 1
    common_neighbours = 2
    jaccard_neighbours = 3
    jaccard_problems = 4
    adar_adamic = 5
    preferential = 6


class VotingStrategy:
    simple = 1
    weighted = 2
    positional = 3


class Engine(object):
    class WeightCalculator(object):
        def __init__(self, engine):
            self.engine = engine

        @staticmethod
        def calc_simple_voting(user, similar_user, sim_value):
            return 1

        def calc_weighted_voting(self, user, similar_user, sim_value):
            return float(sim_value) / reduce(lambda acc, item: acc + item[1], self.engine.similarity[user], 0)

        def calc_positional_voting(self, user, similar_user, sim_value):
            return 1.0 / (self.engine.similarity[user].index((similar_user, sim_value)) + 1)

    class SimilarityCalculator:
        def __init__(self, engine):
            self.engine = engine

        def calc_jaccard_neighbours(self, user1, user2):
            user1_neighbours = self.engine.users_projection_matrix[user1].keys()
            user2_neighbours = self.engine.users_projection_matrix[user2].keys()
            intersection_val = CustomLib.intersection_length(user1_neighbours, user2_neighbours)
            union_val = CustomLib.union_length(user1_neighbours, user2_neighbours)
            ans = 0 if union_val == 0 else intersection_val / union_val
            # if ans != 0.9354838709677419:
            #     CustomLib.debug_print("Jaccard Neighbours Test", user1, self.users[user1], users_projection_matrix[user1], user2, self.users[user2], users_projection_matrix[user2], intersection_val, union_val, ans)
            return ans

        def calc_jaccard_problems(self, user1, user2):
            if user2 not in self.engine.users_projection_matrix[user1]:
                return 0
            intersection_val = self.engine.users_projection_matrix[user1][user2]
            union_val = CustomLib.union_length(self.engine.users[user1], self.engine.users[user2])
            ans = 0 if union_val == 0 else intersection_val / union_val
            # CustomLib.debug_print("Jaccard Problems Test", user1, user2, ans)
            return ans

        def calc_edge_weight(self, user1, user2):
            return 0 if user2 not in self.engine.users_projection_matrix[user1].keys() \
                else self.engine.users_projection_matrix[user1][user2]

        def calc_common_neighbours(self, user1, user2):
            intersection = CustomLib.intersection(self.engine.users_projection_matrix[user1],
                                                  self.engine.users_projection_matrix[user2])
            ans = reduce(lambda acc, user: acc + self.engine.users_projection_matrix[user1][user] +
                                           self.engine.users_projection_matrix[user2][user], intersection, 0)
            return ans

        def calc_adar_atamic(self, user1, user2):
            user1_neighbourhood = self.engine.users_projection_matrix[user1]
            user2_neighbourhood = self.engine.users_projection_matrix[user2]
            intersection = CustomLib.intersection(user1_neighbourhood,
                                                  user2_neighbourhood)
            ans = reduce(lambda acc, user: acc + (user1_neighbourhood[user] + user2_neighbourhood[user])
                                           / math.log(1 + reduce(lambda acc2, i: acc2 + i,
                                                                 self.engine.users_projection_matrix[user].values(),
                                                                 0)
                                                      ),
                         intersection,
                         0)
            # CustomLib.debug_print("Adar Atamic Testing", user1, users[user1], users_projection_matrix[user1], user2, users[user2], users_projection_matrix[user2], intersection, ans)
            return ans

        def calc_preferential(self, user1, user2):
            ans = reduce(lambda acc, i: acc + i, self.engine.users_projection_matrix[user1].values(), 0) * \
                  reduce(lambda acc, i: acc + i, self.engine.users_projection_matrix[user2].values(), 0)
            # CustomLib.debug_print("Preferential Strategy Test", user1, users[user1], users_projection_matrix[user1], user2, users[user2], users_projection_matrix[user2], ans)
            return ans

    users = dict()
    users_projection_matrix = dict()
    similarity = dict()
    recommendations = dict()
    problems = set()
    users_test = dict()
    f1_agg = list()
    recall_agg = list()
    p_agg = list()
    edge_weight_threshold = 0
    similarity_threshold = 0
    test_solve_requirement = 5
    neighbourhood_size = 100
    recommendation_size = 0
    voting_strategy = VotingStrategy.positional
    similarity_strategy = SimilarityStrategy.jaccard_neighbours

    def __init__(self):
        self.weight_calculator = self.WeightCalculator(self)
        self.similarity_calculator = self.SimilarityCalculator(self)

    def clear(self):
        self.users = dict()
        self.users_projection_matrix = dict()
        self.similarity = dict()
        self.recommendations = dict()
        self.problems = set()
        self.users_test = dict()

    def initialize(self, path, recommendation_size):
        self.recommendation_size = recommendation_size
        df = pd.read_csv(path,
                         header=0,
                         names=['id', 'user', 'status'])
        # print(df)
        for row in df.iterrows():
            prob_id = row[1][0]
            user = row[1][1]
            status = row[1][2]
            # print(tempProbId, tempUser, tempStatus)
            if status == 1:
                if user not in self.users.keys():
                    self.users[user] = list()
                self.users[user].append(prob_id)
                self.problems.add(prob_id)

    def initialize_for_test(self, path, recommendation_size, test_size, solve_requirement):
        self.clear()
        self.recommendation_size = recommendation_size
        # CustomLib.debug_print("initialize test", path, test_size, solve_requirement)
        df = pd.read_csv(path,
                         nrows=test_size,
                         header=0,
                         names=['id', 'user', 'status'])
        # print(df)
        for row in df.iterrows():
            prob_id = row[1][0]
            user = row[1][1]
            status = row[1][2]
            # print(tempProbId, tempUser, tempStatus)
            if status == 1:
                if user not in self.users.keys():
                    self.users[user] = list()
                self.users[user].append(prob_id)
                self.problems.add(prob_id)
        df = pd.read_csv(path, skiprows=test_size,
                         header=0, names=['id', 'user', 'status'])
        # print(df)
        for row in df.iterrows():
            prob_id = row[1][0]
            user = row[1][1]
            status = row[1][2]
            # print(tempProbId, tempUser, tempStatus)
            if user in self.users.keys() and status == 1:  # and prob_id in problems:
                if user not in self.users_test.keys():
                    self.users_test[user] = set()
                self.users_test[user].add(prob_id)
        # pprint.pprint(self.users_test)
        delete_items = list()
        for user in self.users.keys():
            if user in self.users_test.keys():
                if len(self.users[user]) < solve_requirement or len(self.users_test[user]) < solve_requirement:
                    delete_items.append(user)
            else:
                delete_items.append(user)
        for i in delete_items:
            self.users.pop(i, 0)
            if i in self.users_test:
                self.users_test.pop(i, 0)

    def build_user_projection_matrix(self):
        for i in self.users:
            if i not in self.users_projection_matrix:
                self.users_projection_matrix[i] = dict()
            for j in self.users:
                if i != j:
                    edge_weight = CustomLib.intersection_length(self.users[i], self.users[j])
                    if edge_weight > self.edge_weight_threshold:
                        self.users_projection_matrix[i][j] = edge_weight

    def get_similarity_value(self, user1, user2):
        solutions = {
            SimilarityStrategy.jaccard_neighbours: self.similarity_calculator.calc_jaccard_neighbours,
            SimilarityStrategy.jaccard_problems: self.similarity_calculator.calc_jaccard_problems,
            SimilarityStrategy.preferential: self.similarity_calculator.calc_preferential,
            SimilarityStrategy.common_neighbours: self.similarity_calculator.calc_common_neighbours,
            SimilarityStrategy.edge_weight: self.similarity_calculator.calc_edge_weight,
            SimilarityStrategy.adar_adamic: self.similarity_calculator.calc_adar_atamic
        }
        if self.similarity_strategy not in solutions:
            raise Exception("Not a viable strategy")
        return solutions[self.similarity_strategy](user1, user2)

    def build_similarity_matrix(self):
        for i in self.users:
            if i not in self.similarity.keys():
                self.similarity[i] = list()
            for j in self.users:
                if i != j:
                    sim_value = self.get_similarity_value(i, j)
                    # print("Entry: {} {} {} {} {}".format(self.users[i], self.users[j], i, j, sim_value))
                    if not set(self.users[j]).issubset(self.users[i]) and sim_value > self.similarity_threshold:
                        self.similarity[i].append((j, sim_value))
                    # else:
                    # print("Subset: {} {} {}".format(self.users[i], self.users[j], sim_value))
        for i in self.similarity:
            self.similarity[i].sort(key=lambda a: a[1], reverse=True)
            # pprint.pprint(self.similarity[i])
            self.similarity[i] = CustomLib.first_k_elements(self.similarity[i], self.neighbourhood_size)
            # [similarity[i][j] for j in range(len(similarity[i])) if j < TOP_K]

    def get_weight_value(self, user, similar_user, sim_value):
        solutions = {
            VotingStrategy.simple: self.weight_calculator.calc_simple_voting,
            VotingStrategy.weighted: self.weight_calculator.calc_weighted_voting,
            VotingStrategy.positional: self.weight_calculator.calc_positional_voting
        }
        if self.voting_strategy not in solutions:
            raise Exception("Not a viable voting strategy")
        return solutions[self.voting_strategy](user, similar_user, sim_value)

    def build_recommendation_matrix(self):
        for i in self.users.keys():
            if i not in self.recommendations.keys():
                self.recommendations[i] = defaultdict(float)
            for (userID, simValue) in self.similarity[i]:
                for prob in self.users[userID]:
                    if prob not in self.users[i]:
                        self.recommendations[i][prob] += self.get_weight_value(i, userID, simValue)
            # CustomLib.debug_print("Recommendations", i, self.recommendations[i], self.similarity[i])
        # for i in self.users:
        #     CustomLib.debug_print("Recommendations values test", i, self.similarity[i], recommendations[i])
        # for (j, sim_value) in self.similarity[i]:
        #     CustomLib.debug_print("Values in self.similarity", j, sorted(self.users[j]), sim_value)
        # exit(0)
        for i in self.recommendations:
            temp = list(self.recommendations[i].items())
            temp.sort(key=lambda a: a[1], reverse=True)
            temp = CustomLib.first_k_elements(temp, self.recommendation_size)
            # print("{}: {}".format(i, temp))
            self.recommendations[i] = dict()
            for (prob, count) in temp:
                self.recommendations[i][prob] = count

    def perform_test(self):
        precision = 0
        recall = 0
        count = 0
        for (username, problemsSolved) in self.users_test.items():
            if username in self.recommendations.keys() and len(self.recommendations[username].keys()) > 0:
                # if CustomLib.intersection(recommendations[username].keys(), problemsSolved) != len(problemsSolved):
                #     CustomLib.debug_print("Testing Solution", sorted(recommendations[username].keys()), sorted(problemsSolved))
                count += 1
                true_positive = 0
                # print(recommendations[username].keys())
                # print("{} : {} - {}".format(username, recommendations[username], problemsSolved))
                for probId in problemsSolved:
                    if probId in self.recommendations[username].keys():
                        # print(username, probId)
                        true_positive += 1
                precision += true_positive / len(self.recommendations[username].keys())
                recall += true_positive / len(problemsSolved)
        # print(precision, count)
        if count != 0:
            precision = precision / count
            recall = recall / count
            f1 = 2 * precision * recall / (precision + recall)
        else:
            precision = recall = f1 = 0
        self.p_agg.append(precision)
        self.recall_agg.append(recall)
        self.f1_agg.append(f1)
        print("Precision = {}, Recall = {}, F1 = {}, Count = {}".format(precision, recall, f1, count))
        # for i in recommendations:
        #     print(i, len(recommendations[i]), recommendations[i])

    def print_means(self):
        print("Precision: {}\nRecall: {}\nF1: {}".format(
            statistics.mean(self.p_agg),
            statistics.mean(self.recall_agg),
            statistics.mean(self.f1_agg)))
