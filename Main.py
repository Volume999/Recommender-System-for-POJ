from Engine import Engine

TOP_N = [3, 5, 7, 10, 20]
TEST_SAMPLE = [200, 300, 400, 500, 600]
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
        engine.execute()
        engine.perform_test()
    engine.print_means()
    engine.full_clear()
