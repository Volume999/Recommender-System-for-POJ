from Enums import VotingStrategy, SimilarityStrategy


class Variables:
    edge_weight_threshold = 2
    similarity_threshold = 0
    neighbourhood_size = 100
    recommendation_size = 5
    voting_strategy = VotingStrategy.weighted
    similarity_strategy = SimilarityStrategy.jaccard_neighbours
    user_solve_requirements = 4
    path = str()