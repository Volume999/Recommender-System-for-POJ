import statistics
from collections import defaultdict
import pandas as pd
import CustomLib
from CustomLib import UserTypes, SubmissionType, VerdictTypes, ProblemDifficulty, VotingStrategy, SimilarityStrategy
import random
from Calculators import WeightCalculator, SimilarityCalculator, calc_edge
import csv


class Engine:
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
            self.unsolved_threshold = 0
            self.solved_threshold = 0
            self.problem_type = ProblemDifficulty.variable

    class Variables:
        edge_weight_threshold = 2
        similarity_threshold = 0
        neighbourhood_size = 100
        recommendation_size = 15
        voting_strategy = VotingStrategy.weighted
        similarity_strategy = SimilarityStrategy.jaccard_neighbours
        user_solve_requirements = 4
        path = str()

    class Data:
        def __init__(self):
            self.path = str()
            self.users = dict()
            self.problems = dict()

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
            f1 = 2 * precision * recall / (precision + recall)
            self.p_agg.append(precision)
            self.recall_agg.append(recall)
            self.f1_agg.append(f1)
            self.one_hit_agg.append(one_hit)
            self.mrr_agg.append(mrr)

        def print_means(self):
            with open(file='stats.csv', mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([self.test_solve_requirement,
                                 self.engine.Variables.recommendation_size,
                                 self.engine.Variables.similarity_strategy,
                                 self.engine.Variables.voting_strategy,
                                 self.engine.Variables.edge_weight_threshold,
                                 self.p_agg,
                                 self.recall_agg,
                                 self.f1_agg,
                                 self.one_hit_agg,
                                 self.mrr_agg])

    def __init__(self, path=""):
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
        df = pd.read_csv(self.data.path,
                         header=0,
                         names=['id', 'user', 'status', 'count'])
        users = self.data.users
        problems = self.data.problems
        for row in df.iterrows():
            prob_id = row[1][0]
            user = str(row[1][1])
            status = row[1][2]
            count = row[1][3]
            if count < 100:
                if user not in users.keys():
                    users[user] = self.User()
                if status == VerdictTypes.success.value:
                    users[user].problems_solved.append(prob_id)
                elif status == VerdictTypes.fail.value or status == VerdictTypes.partially_solved.value:
                    users[user].problems_unsolved.append(prob_id)
                users[user].submissions_stats[prob_id] = self.SubmissionStats(attempts=count)
                if status == VerdictTypes.partially_solved.value:
                    users[user].submissions_stats[prob_id].submission_type = SubmissionType.solved_partially

                if prob_id not in problems:
                    problems[prob_id] = self.ProblemStats()
                if status == VerdictTypes.success.value:
                    problems[prob_id].attempts_before_success.append(count)
                elif status == VerdictTypes.fail.value or status == VerdictTypes.partially_solved.value:
                    problems[prob_id].attempts_before_fail.append(count)
        delete_users = list()
        for user in users:
            if len(users[user].problems_solved) < self.Variables.user_solve_requirements:
                delete_users.append(user)
        for user in delete_users:
            users.pop(user)

    def initialize_tests(self):
        users = self.data.users
        for user in users:
            if len(users[user].problems_solved) >= self.testing.test_solve_requirement * 2:
                self.testing.users_test[user] = set()
                random.Random(228).shuffle(users[user].problems_solved)
                users[user].problems_solved, self.testing.users_test[user] = CustomLib.split_in_half(
                    users[user].problems_solved)
                for prob in self.testing.users_test[user]:
                    users[user].submissions_stats.pop(prob)

    def categorize_problems(self):
        problems = self.data.problems
        for prob in problems:
            problems[prob].solved_threshold = 0 \
                if len(problems[prob].attempts_before_success) == 0 \
                else statistics.mean(problems[prob].attempts_before_success)
            problems[prob].unsolved_threshold = 0 if len(problems[prob].attempts_before_fail) == 0 \
                else statistics.mean(problems[prob].attempts_before_fail)
            solved_with_many = len([val for val in problems[prob].attempts_before_success if
                                    val > problems[prob].solved_threshold])
            solved_with_little = len([val for val in problems[prob].attempts_before_success if
                                      val <= problems[prob].solved_threshold])
            if len(problems[prob].attempts_before_success) > 1:
                if solved_with_little >= 2 * solved_with_many:
                    problems[prob].problem_type = ProblemDifficulty.easy
                elif solved_with_many >= 2 * solved_with_little:
                    problems[prob].problem_type = ProblemDifficulty.difficult
                else:
                    problems[prob].problem_type = ProblemDifficulty.variable

    def categorize_user(self, user):
        problems = self.data.problems
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
        if solved_with_many >= 2 * solved_with_few:
            user.user_type = UserTypes.imprecise
        elif solved_with_few >= 2 * solved_with_many:
            user.user_type = UserTypes.precise
        else:
            user.user_type = UserTypes.variable

    def categorize_users(self):
        for user in self.data.users:
            self.categorize_user(self.data.users[user])

    def manage_noise_user(self, user):
        problems = self.data.problems
        for prob in user.problems_solved:
            if problems[prob].problem_type.value == user.user_type.value \
                    and problems[prob].problem_type != ProblemDifficulty.variable:
                user.submissions_stats[prob].submission_type = \
                    SubmissionType.solved_with_few \
                    if user.user_type == UserTypes.precise \
                    else SubmissionType.solved_with_many

    def manage_noise_users(self):
        for user in self.data.users:
            self.manage_noise_user(self.data.users[user])

    def get_user_projections(self, user):
        ans = dict()
        for j in self.data.users:
            user2 = self.data.users[j]
            if user is not user2:
                edge_weight = calc_edge(user, user2)
                if edge_weight > self.Variables.edge_weight_threshold:
                    ans[user2] = edge_weight
        return ans

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

    def get_user_similarities(self, user):
        ans = list()
        for j in self.data.users:
            user2 = self.data.users[j]
            if user is not user2:
                sim_value = self.get_similarity_value(user, user2)
                if not set(user2.problems_solved).issubset(user.problems_solved) \
                        and sim_value > self.Variables.similarity_threshold:
                    ans.append((user2, sim_value))
        ans.sort(key=lambda a: a[1], reverse=True)
        # ans = CustomLib.first_k_elements(ans, self.Variables.neighbourhood_size)
        return ans[:self.Variables.neighbourhood_size]

    def build_similarities(self):
        for i in self.data.users:
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
            for prob in user2.problems_solved:
                if prob not in user.problems_solved:
                    ans[prob] += voting_weight
        temp = list(ans.items())
        temp.sort(key=lambda a: a[1], reverse=True)
        # temp = CustomLib.first_k_elements(temp, self.Variables.recommendation_size)
        return temp[:self.Variables.recommendation_size]

    def fill_recommendations(self, user, problems):
        i = 0
        while len(user.recommendations) < self.Variables.recommendation_size and i < len(problems):
            user.recommendations.append((problems[i][0], 0))
            i += 1

    def build_recommendations(self):
        easy_problems = list(filter(lambda prob: prob[1].problem_type == ProblemDifficulty.easy,
                                    self.data.problems.items()))
        easy_problems.sort(key=lambda item: len(item[1].attempts_before_success) + len(item[1].attempts_before_fail),
                           reverse=True)
        for i in self.data.users.keys():
            self.data.users[i].recommendations = self.get_user_recommendations(self.data.users[i])
            if len(self.data.users[i].recommendations) < self.Variables.recommendation_size:
                self.fill_recommendations(self.data.users[i], easy_problems)

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
                    self.testing.clear_aggregates()
                    self.initialize()
                    self.initialize_tests()
                    self.execute()
                    self.testing.perform_test()
                    self.testing.print_means()
        else:
            self.Variables.similarity_strategy = SimilarityStrategy.jaccard_neighbours
            self.Variables.voting_strategy = VotingStrategy.weighted
            self.testing.clear_aggregates()
            self.initialize()
            self.initialize_tests()
            self.execute()
            self.testing.perform_test()
            self.testing.print_means()
            print(len(self.data.users), len(self.testing.users_test))
            # for us in self.data.users:
            #     user = self.data.users[us]
            #     print(us, len(user.recommendations), len(user.similarities))
            #     if us in ['monich', 'dasturchi2018', 'Test777', 'chorobaev']:
            #         print(us,
            #               user.recommendations)
