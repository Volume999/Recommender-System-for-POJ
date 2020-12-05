import csv
import random
import CustomLib
from Variables import Variables
import statistics

# Logic for testing
class Testing:
    # You need to have 2 * test_solve_requirements amounts of problems solved
    # to be considered for recommendations
    test_solve_requirement = 5

    # TO DO: CHECK IF f1_agg, recall_agg, p_agg, one_hit_agg, mrr_agg
    # need to be lists or should be
    def __init__(self, engine):
        self.one_hit = 0
        self.recall = 0
        self.precision = 0
        self.mrr = 0
        self.f1 = 0
        self.engine = engine
        self.users_test = dict()
        self.f1_agg = list()
        self.recall_agg = list()
        self.p_agg = list()
        self.one_hit_agg = list()
        self.mrr_agg = list()

    # Performing tests functionality
    def perform_test(self):
        users = self.engine.data.users
        tests = self.users_test
        for (username, problemsSolved) in tests.items():
            user_recommendations = [prob[0] for prob in users[username].recommendations]
            if len(user_recommendations) > 0:
                # true positive - number of problems that were recommended,
                # that the person eventually solved
                true_positive = 0
                for probId in problemsSolved:
                    if probId in user_recommendations:
                        true_positive += 1
                # Precision - true positive divided by number
                # of recommended items
                # Recall - ratio of true positive to number of items
                # the user solved eventually
                self.precision += true_positive / len(user_recommendations)
                self.recall += true_positive / len(problemsSolved)
                # One hit - at least one task that the user solved eventually
                # was recommended
                if true_positive > 0:
                    self.one_hit += 1
        # MRR - Mean reciprocal rank and its calculation
        mrr_list = list()
        for j in tests:
            user = users[j]
            user_r = [prob[0] for prob in user.recommendations]
            for i in range(len(user_r)):
                prob = list(user_r)[i]
                if prob in tests[j]:
                    mrr_list.append(1 / (i + 1))
                    break
        self.mrr = statistics.mean(mrr_list) if len(mrr_list) > 0 else 0
        self.precision /= len(tests)
        self.recall /= len(tests)
        self.one_hit /= len(tests)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall != 0 else 0

    def print_results(self):
        # Printing the results
        with open(file='stats.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([self.test_solve_requirement,
                             Variables.recommendation_size,
                             Variables.similarity_strategy,
                             Variables.voting_strategy,
                             Variables.edge_weight_threshold,
                             self.precision,
                             self.recall,
                             self.f1,
                             self.one_hit,
                             self.mrr])

    def initialize_tests(self):
        self.users_test = dict()
        users = self.engine.data.users
        # Tests initialization - for each user that qualifies for testing
        # Divide his submissions in half, use one for training and one for
        # Testing
        for user in users:
            if len(users[user].problems_solved) >= self.test_solve_requirement * 2:
                self.users_test[user] = set()
                random.Random(228).shuffle(users[user].problems_solved)
                users[user].problems_solved, self.users_test[user] = CustomLib.split_in_half(
                    users[user].problems_solved)
                for prob in self.users_test[user]:
                    users[user].submissions_stats.pop(prob)
