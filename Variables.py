from Enums import VotingStrategy, SimilarityStrategy


class Variables:
    # Edge weight threshold - minimum edge weight to be considered for
    # neighbourhood
    edge_weight_threshold = 2
    # Similarity threshold - minimum similarity value to be considered
    # for neighbourhood
    similarity_threshold = 0
    # neighbourhood - people that are considered for building the
    # recommendation list
    neighbourhood_size = 100
    # Recommendation size - size of the recommendation list
    recommendation_size = 5
    # Voting strategy - considering the neighbours and their edges
    # when building the recommendation list
    voting_strategy = VotingStrategy.weighted
    # Similarity strategy - the algorithm to calculate the
    # Similarity
    similarity_strategy = SimilarityStrategy.jaccard_neighbours
    # User solve requirements - required number of tasks to be solved
    # to be considered for analysis and testing
    user_solve_requirements = 4
    # path - path to CSV
    path = str()
    #engine_pickle_file_name - file name of saved engine
    engine_pickle_file_name_5 = 'Engine.pickle'
    engine_pickle_file_name_15 = 'Engine2.pickle'
