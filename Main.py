from Engine import Engine

TOP_N = [10]
test_solve_requirement = 3

engine = Engine(path="/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions3.csv")
engine.test_solve_requirement = test_solve_requirement
for N in TOP_N:
    print("N = {}".format(N))
    engine.recommendation_size = N
    engine.test()
