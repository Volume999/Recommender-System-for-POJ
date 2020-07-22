import math
from functools import reduce

import CustomLib


class WeightCalculator:
    @staticmethod
    def calc_simple_voting(user, similar_user, sim_value):
        return 1

    @staticmethod
    def calc_weighted_voting(user, similar_user, sim_value):
        similarity_sum = reduce(lambda acc, item: acc + item[1], user.similarities, 0)
        # CustomLib.debug_print("weighted", user, sim_value, sum)
        return float(sim_value) / similarity_sum

    @staticmethod
    def calc_positional_voting(user, similar_user, sim_value):
        return 1.0 / (user.similarities.index((similar_user, sim_value)) + 1)


class SimilarityCalculator:
    @staticmethod
    def calc_jaccard_neighbours(user1, user2):
        # user1_neighbours = self.engine.data.users_projection_matrix[user1].keys()
        # user2_neighbours = self.engine.data.users_projection_matrix[user2].keys()
        user1_neighbours = list(user1.projections.keys())
        user2_neighbours = list(user2.projections.keys())
        intersection_val = CustomLib.intersection_length(user1_neighbours, user2_neighbours)
        union_val = CustomLib.union_length(user1_neighbours, user2_neighbours)
        ans = 0 if union_val == 0 else intersection_val / union_val
        return ans

    @staticmethod
    def calc_jaccard_problems(user1, user2):
        if user2 not in user1.projections:
            # CustomLib.debug_print("Jaccard check", )
            return 0
        intersection_val = user1.projections[user2]
        union_val = CustomLib.union_length(user1.problems_solved + user1.problems_unsolved,
                                           user2.problems_solved + user2.problems_unsolved)
        ans = 0 if union_val == 0 else intersection_val / union_val
        return ans

    @staticmethod
    def calc_edge_weight(user1, user2):
        return 0 if user2 not in user1.projections \
            else user1.projections[user2]

    @staticmethod
    def calc_common_neighbours(user1, user2):
        intersection = CustomLib.intersection(user1.projections, user2.projections)
        ans = reduce(lambda acc, user: acc + user1.projections[user] + user2.projections[user], intersection, 0)
        return ans

    @staticmethod
    def calc_adar_atamic(user1, user2):
        user1_neighbourhood = user1.projections
        user2_neighbourhood = user2.projections
        intersection = CustomLib.intersection(user1_neighbourhood,
                                              user2_neighbourhood)
        ans = reduce(lambda acc, user: acc + (user1_neighbourhood[user] + user2_neighbourhood[user])
                                       / math.log(1 + reduce(lambda acc2, i: acc2 + i,
                                                             user.projections.values(),
                                                             0)
                                                  ),
                     intersection,
                     0)
        return ans

    @staticmethod
    def calc_preferential(user1, user2):
        ans = reduce(lambda acc, i: acc + i, user1.projections.values(), 0) * \
              reduce(lambda acc, i: acc + i, user2.projections.values(), 0)
        return ans