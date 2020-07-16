from Engine import Engine

TOP_N = [10]
engine = Engine(path="/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions3.csv")
user = engine.User
user.problems_solved = [2, 10, 12]
user.problems_unsolved = [43, 44, 49]
user.submissions_stats = {
    2: 1,
    10: 2,
    12: 3,
    43: 4,
    44: 5,
    49: 6
}

for N in TOP_N:
    print("N = {}".format(N))
    engine.recommendation_size = N
    engine.test()