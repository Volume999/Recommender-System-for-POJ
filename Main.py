from Engine import Engine

TOP_N = [5, 10, 15, 20, 30, 40, 50]
engine = Engine(path="/Users/citius/Desktop/Study/SeniorThesisWork/OlympSubmissions.csv")
# user = engine.User
# user.problems_solved = [2, 10, 12]
# user.problems_unsolved = [43, 44, 49]
# user.submissions_stats = {
#     2: 1,
#     10: 2,
#     12: 3,
#     43: 4,
#     44: 5,
#     49: 6
# }

# user_data = [(123, 1, 1,), (124, 1, 1), (12312, 1, 1), (412321, 1, 1)]

user_data = [(1, 1, 1),
             (10, 1, 2),
             (19, 1, 3),
             (60, 1, 4),
             (59, 2, 4),
             (80, 2, 5),
             ]

for N in TOP_N:
    # print("N = {}".format(N))
    for t in range(15):
        engine.Variables.edge_weight_threshold = t
        engine.Variables.recommendation_size = N
        engine.test()
# engine.run()
# engine.execute_for_user(user_data)
