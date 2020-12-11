from Engine import Engine

# Recommendation list size

TOP_N = [15]


# Engine initialization

engine = Engine(path="/Users/citius/Desktop/Study/SeniorThesisWork/OlympSubmissions.csv")

# <explanation>
# Testing Recommendations for a user
# Input - Solved problems, unsolved problems and their statistics ( number of attempts per problem )
# </explanation>

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

# <explanation>
# Testing Recommendations for a user
# Input - Solved problems, unsolved problems and their statistics ( number of attempts per problem )
# Input - list of tuples
# tuples of type (task ID,
#                 status (1 - solved, 2 - unsolved),
#                 number of attempts (until first solved or total if unsolved)
# </explanation>
user_data = [(1, 1, 1),
             (10, 1, 2),
             (19, 1, 3),
             (60, 1, 4),
             (59, 2, 4),
             (80, 2, 5),
             ]

# engine.test to initialize engine and test it
# engine.run to initialize and leave ready for input
# engine.test()
engine.run()
print(engine.data.users.keys())
# print('start')
# engine.execute_for_user(user_data)
