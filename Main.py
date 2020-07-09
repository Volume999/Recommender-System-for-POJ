from Engine import Engine

TOP_N = [3, 5]
TEST_SAMPLE = [300, 400, 500]
test_solve_requirement = 5

engine = Engine()
for test_sample in TEST_SAMPLE:
    print("Test_sample = {}, voting strategy = {}, similarity strategy = {}".format(test_sample,
                                                                                    engine.voting_strategy,
                                                                                    engine.similarity_strategy))
    for N in TOP_N:
        print("N = {}".format(N))
        path = "/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions3.csv"
        engine.initialize_for_test(path, N, test_sample, test_solve_requirement)
        engine.categorize_problems()
        engine.build_user_projection_matrix()
        engine.build_similarity_matrix()
        engine.build_recommendation_matrix()
        engine.perform_test()
engine.print_means()
