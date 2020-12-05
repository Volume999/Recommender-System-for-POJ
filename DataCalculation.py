from Calculators import calc_edge, SimilarityCalculator, WeightCalculator
from Enums import VotingStrategy, SimilarityStrategy, ProblemDifficulty
from Variables import Variables
from collections import defaultdict


class Calculator:
    def __init__(self, data):
        self.users = data.users
        self.problems = data.problems

    # User projections - number of tasks that both users A and B solved or
    # unsolved with a similar or same submissions status
    def get_user_projections(self, user):
        ans = dict()
        for j in self.users:
            user2 = self.users[j]
            if user is not user2:
                edge_weight = calc_edge(user, user2)
                if edge_weight > Variables.edge_weight_threshold:
                    ans[user2] = edge_weight
        return ans

    def build_user_projections(self):
        for i in self.users.keys():
            self.users[i].projections = self.get_user_projections(self.users[i])

    def get_similarity_value(self, user1, user2):
        solutions = {
            SimilarityStrategy.jaccard_neighbours: SimilarityCalculator.calc_jaccard_neighbours,
            SimilarityStrategy.jaccard_problems: SimilarityCalculator.calc_jaccard_problems,
            SimilarityStrategy.preferential: SimilarityCalculator.calc_preferential,
            SimilarityStrategy.common_neighbours: SimilarityCalculator.calc_common_neighbours,
            SimilarityStrategy.edge_weight: SimilarityCalculator.calc_edge_weight,
            SimilarityStrategy.adar_adamic: SimilarityCalculator.calc_adar_atamic
        }
        if Variables.similarity_strategy not in solutions:
            raise Exception("Not a viable strategy")
        return solutions[Variables.similarity_strategy](user1, user2)

    def get_user_similarities(self, user):
        ans = list()
        for j in self.users:
            user2 = self.users[j]
            if user is not user2:
                sim_value = self.get_similarity_value(user, user2)
                if not set(user2.problems_solved).issubset(user.problems_solved) \
                        and sim_value > Variables.similarity_threshold:
                    ans.append((user2, sim_value))
        ans.sort(key=lambda a: a[1], reverse=True)
        # ans = CustomLib.first_k_elements(ans, Variables.neighbourhood_size)
        return ans[:Variables.neighbourhood_size]

    def build_similarities(self):
        for i in self.users:
            self.users[i].similarities = self.get_user_similarities(self.users[i])

    def get_weight_value(self, user, similar_user, sim_value):
        solutions = {
            VotingStrategy.simple: WeightCalculator.calc_simple_voting,
            VotingStrategy.weighted: WeightCalculator.calc_weighted_voting,
            VotingStrategy.positional: WeightCalculator.calc_positional_voting
        }
        if Variables.voting_strategy not in solutions:
            raise Exception("Not a viable voting strategy")
        return solutions[Variables.voting_strategy](user, similar_user, sim_value)

    def get_user_recommendations(self, user):
        ans = defaultdict(float)
        for (user2, simValue) in user.similarities:
            voting_weight = self.get_weight_value(user, user2, simValue)
            for prob in user2.problems_solved:
                if prob not in user.problems_solved:
                    ans[prob] += voting_weight
        # sorts the list by voting weight, then selects N problems and returns as list (probId, weight)
        return sorted(list(ans.items()), key=lambda item: item[1], reverse=True)[:Variables.recommendation_size]

    def build_recommendations(self):
        # Easy problems - problem-set used to fill recommendations
        # Problems categorized as Easy, and sorted by popularity
        easy_problems = list(filter(lambda prob: prob[1].problem_type == ProblemDifficulty.easy,
                                    self.problems.items()))
        easy_problems.sort(key=lambda item: len(item[1].attempts_before_success) + len(item[1].attempts_before_fail),
                           reverse=True)
        for i in self.users.keys():
            self.users[i].recommendations = self.get_user_recommendations(self.users[i])
            dif = Variables.recommendation_size - len(self.users[i].recommendations)
            if dif > 0:
                # if recommendation list is smaller than needed, add all easy problems that user did not solve
                self.users[i].recommendations.extend(easy_problems[:dif] if dif < len(easy_problems) else easy_problems)

    # Calculator logic:
    # 1. Build user - projections
    # 2. Build similarity matrix
    # 3. Build recommendation list
    def calculate(self):
        self.build_user_projections()
        self.build_similarities()
        self.build_recommendations()

    def calculate_user(self, user):
        self.get_user_projections(user)
        self.get_user_similarities(user)
        self.get_user_recommendations(user)