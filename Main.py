import pprint
import csv
from functools import reduce
import pandas as pd
import statistics
from CustomLib import CustomLib
import math


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


TOP_K = 30
TOP_N = [1, 3, 5]
TEST_SAMPLE = [300, 400, 500]
SOLVE_REQUIREMENT = 5
SIMILARITY_THRESHOLD_VALUE = 0
EDGE_WEIGHT_THRESHOLD_VALUE = 0
users = dict()
users_projection_matrix = dict()
similarity = dict()
recommendations = dict()
problems = set()
users_test = dict()
f1_agg = list()
recall_agg = list()
p_agg = list()


class SimilarityCalculator:
    def __init__(self, strategy):
        self.strategy = strategy

    @staticmethod
    def calc_jaccard_neighbours(user1, user2):
        user1_neighbours = users_projection_matrix[user1].keys()
        user2_neighbours = users_projection_matrix[user2].keys()
        intersection_val = CustomLib.intersection_length(user1_neighbours, user2_neighbours)
        union_val = CustomLib.union_length(user1_neighbours, user2_neighbours)
        ans = 0 if union_val == 0 else intersection_val / union_val
        # if ans != 0.9354838709677419:
        #     CustomLib.debug_print("Jaccard Neighbours Test", user1, users[user1], users_projection_matrix[user1], user2, users[user2], users_projection_matrix[user2], intersection_val, union_val, ans)
        return ans

    @staticmethod
    def calc_jaccard_problems(user1, user2):
        if user2 not in users_projection_matrix[user1]:
            return 0
        intersection_val = users_projection_matrix[user1][user2]
        union_val = CustomLib.union_length(users[user1], users[user2])
        ans = 0 if union_val == 0 else intersection_val / union_val
        # CustomLib.debug_print("Jaccard Problems Test", user1, user2, ans)
        return ans

    @staticmethod
    def calc_edge_weight(user1, user2):
        return 0 if user2 not in users_projection_matrix[user1].keys() \
            else users_projection_matrix[user1][user2]

    @staticmethod
    def calc_common_neighbours(user1, user2):
        intersection = CustomLib.intersection(users_projection_matrix[user1],
                                              users_projection_matrix[user2])
        ans = reduce(lambda acc, user: acc + users_projection_matrix[user1][user] + users_projection_matrix[user2][user], intersection, 0)
        return ans

    @staticmethod
    def calc_adar_atamic(user1, user2):
        intersection = CustomLib.intersection(users_projection_matrix[user1],
                                              users_projection_matrix[user2])
        ans = reduce(lambda acc, user: acc +
                                       (users_projection_matrix[user1][user] +
                                        users_projection_matrix[user2][user]) /
                                       math.log(1 + reduce(lambda acc2,
                                                                  i: acc2 + i,
                                                           users_projection_matrix[user].values(),
                                                           0)),
                     intersection,
                     0)
        # CustomLib.debug_print("Adar Atamic Testing", user1, users[user1], users_projection_matrix[user1], user2, users[user2], users_projection_matrix[user2], intersection, ans)
        return ans

    @staticmethod
    def calc_preferential(user1, user2):
        ans = reduce(lambda acc, i: acc + i, users_projection_matrix[user1].values(), 0) * \
              reduce(lambda acc, i: acc + i, users_projection_matrix[user2].values(), 0)
        # CustomLib.debug_print("Preferential Strategy Test", user1, users[user1], users_projection_matrix[user1], user2, users[user2], users_projection_matrix[user2], ans)
        return ans

    def get_similarity_value(self, user1, user2):
        if self.strategy == SimilarityStrategy.jaccard_neighbours:
            return self.calc_jaccard_neighbours(user1, user2)
        if self.strategy == SimilarityStrategy.jaccard_problems:
            return self.calc_jaccard_problems(user1, user2)
        if self.strategy == SimilarityStrategy.edge_weight:
            return self.calc_edge_weight(user1, user2)
        if self.strategy == SimilarityStrategy.common_neighbours:
            return self.calc_common_neighbours(user1, user2)
        if self.strategy == SimilarityStrategy.adar_adamic:
            return self.calc_adar_atamic(user1, user2)
        if self.strategy == SimilarityStrategy.preferential:
            return self.calc_preferential(user1, user2)
        raise Exception("Not a viable strategy")


similarity_strategy = SimilarityStrategy.edge_weight
similarity_calculator = SimilarityCalculator(similarity_strategy)

# users = dictionary, key - userID value = vector of problems accepted
# similarity = dictionary, key - userID value - tuples of other userID and its similarity value
# recommendations = dictionary, key - userID, value - dictionary, key - problemID, value - its weight


def initialize():
    global users,\
        similarity,\
        recommendations,\
        problems,\
        users_test,\
        f1_agg,\
        recall_agg,\
        p_agg
    users = dict()
    similarity = dict()
    recommendations = dict()
    problems = set()
    users_test = dict()


def build_user_projection_matrix():
    global users_projection_matrix
    for i in users:
        if i not in users_projection_matrix:
            users_projection_matrix[i] = dict()
        for j in users:
            if i != j:
                edge_weight = CustomLib.intersection_length(users[i], users[j])
                if edge_weight > EDGE_WEIGHT_THRESHOLD_VALUE:
                    users_projection_matrix[i][j] = edge_weight


def build_similarity_matrix():
    global similarity
    for i in users:
        if i not in similarity.keys():
            similarity[i] = list()
        for j in users:
            if i != j:
                sim_value = similarity_calculator.get_similarity_value(i, j)
                # print("Entry: {} {} {} {} {}".format(users[i], users[j], i, j, sim_value))
                if not set(users[j]).issubset(users[i]) and sim_value > SIMILARITY_THRESHOLD_VALUE:
                    similarity[i].append((j, sim_value))
                # else:
                # print("Subset: {} {} {}".format(users[i], users[j], sim_value))
    for i in similarity:
        similarity[i].sort(key=lambda a: a[1], reverse=True)
        # pprint.pprint(similarity[i])
        similarity[i] = CustomLib.first_k_elements(similarity[i], TOP_K)
        # [similarity[i][j] for j in range(len(similarity[i])) if j < TOP_K]


def build_recommendation_matrix(n):
    global recommendations
    for i in users.keys():
        if i not in recommendations.keys():
            recommendations[i] = dict()
        for (userID, simValue) in similarity[i]:
            for prob in users[userID]:
                if prob not in users[i]:
                    # print(i, userID, prob)
                    if prob not in recommendations[i].keys():
                        recommendations[i][prob] = simValue
                    else:
                        recommendations[i][prob] += simValue
                    # print("{}: {} {}".format(i, prob, recommendations[i][prob]))
    for i in recommendations:
        temp = list(recommendations[i].items())
        temp.sort(key=lambda a: a[1], reverse=True)
        temp = CustomLib.first_k_elements(temp, n)
        # print("{}: {}".format(i, temp))
        recommendations[i] = dict()
        for (prob, count) in temp:
            recommendations[i][prob] = count


def perform_test():
    global recall_agg, f1_agg, p_agg
    precision = 0
    recall = 0
    count = 0
    for (username, problemsSolved) in users_test.items():
        if username in recommendations.keys() and len(recommendations[username].keys()) > 0:
            # if CustomLib.intersection(recommendations[username].keys(), problemsSolved) != len(problemsSolved):
            #     CustomLib.debug_print("Testing Solution", sorted(recommendations[username].keys()), sorted(problemsSolved))
            count += 1
            truePositive = 0
            # print(recommendations[username].keys())
            falsePositive = len(recommendations[username].keys())
            falseNegative = 0
            trueNegative = 0
            # print("{} : {} - {}".format(username, recommendations[username], problemsSolved))
            for probId in problemsSolved:
                if probId in recommendations[username].keys():
                    # print(username, probId)
                    truePositive += 1
                    falsePositive -= 1
                else:
                    falseNegative += 1
            for prob in recommendations[username]:
                if prob not in problemsSolved:
                    trueNegative += 1
            # print(truePositive, falsePositive)
            # print(truePositive, len(recommendations[username].keys()))
            precision += truePositive / len(recommendations[username].keys())
            recall += truePositive / len(problemsSolved)
    # print(precision, count)
    if count != 0:
        precision = precision / count
        recall = recall / count
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = f1 = 0
    p_agg.append(precision)
    recall_agg.append(recall)
    f1_agg.append(f1)
    print("Precision = {}, Recall = {}, F1 = {}, Count = {}".format(precision, recall, f1, count))
    # for i in recommendations:
    #     print(i, len(recommendations[i]), recommendations[i])

def preparation(TEST):
    global users, users_test, problems
    df = pd.read_csv('/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions2.csv',
                     nrows=TEST, header=0, names=['id', 'user', 'status'])
    # print(df)
    for row in df.iterrows():
        ProbId = row[1][0]
        User = row[1][1]
        Status = row[1][2]
        # print(tempProbId, tempUser, tempStatus)
        if Status == 1:
            if User not in users.keys():
                users[User] = list()
            users[User].append(ProbId)
            problems.add(ProbId)
    df = pd.read_csv('/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions2.csv', skiprows=TEST,
                     header=0, names=['id', 'user', 'status'])
    # print(df)
    for row in df.iterrows():
        ProbId = row[1][0]
        User = row[1][1]
        Status = row[1][2]
        # print(tempProbId, tempUser, tempStatus)
        if User in users.keys() and Status == 1: #and ProbId in problems:
            if User not in users_test.keys():
                users_test[User] = set()
            users_test[User].add(ProbId)
    # pprint.pprint(users_test)
    delete_items = list()
    for user in users.keys():
        if user in users_test.keys():
            if len(users[user]) < SOLVE_REQUIREMENT or len(users_test[user]) < SOLVE_REQUIREMENT:
                delete_items.append(user)
        else:
            delete_items.append(user)
    for i in delete_items:
        users.pop(i, 0)
        if i in users_test:
            users_test.pop(i, 0)


for test in TEST_SAMPLE:
    print("Test_sample = {}".format(test))
    for N in TOP_N:
        print("N = {}".format(N))
        initialize()
        preparation(test)
        build_user_projection_matrix()
        build_similarity_matrix()
        build_recommendation_matrix(N)
        perform_test()
# print(p_agg, recall_agg, f1_agg)
print("Precision: {}\nRecall: {}\nF1: {}".format(
    statistics.mean(p_agg),
    statistics.mean(recall_agg),
    statistics.mean(f1_agg)))
