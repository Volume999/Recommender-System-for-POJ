from Engine import Engine

TOP_N = [10]
engine = Engine(path="/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions3.csv")
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

user_data = [(2, 1, 1),
             (10, 1, 2),
             (12, 1, 3),
             (43, 2, 4),
             (44, 2, 5),
             (49, 2, 6),
             (47, 1, 2),
             (48, 1, 1),
             (39, 1, 1)]

for N in TOP_N:
    print("N = {}".format(N))
    engine.recommendation_size = N
    # engine.test()
    engine.run()
    engine.execute_for_user(user_data)
