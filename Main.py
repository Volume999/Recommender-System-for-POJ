from Engine import get_engine
from DataSource import DataSourceCsv, EnginePickle
from Enums import RunMode
from CustomLib import debug_timing
# Recommendation list size

TOP_N = [15]


# Engine initialization
data_source = DataSourceCsv(file_path="/Users/citius/Desktop/Study/SeniorThesisWork/OlympSubmissions.csv")
engine_source = EnginePickle(file_path="Engine.pickle")
mode = RunMode.run
# end of initialization
engine = get_engine(engine_source, mode)

# <explanation>
# Testing Recommendations for a user
# Input - Solved problems, unsolved problems and their statistics ( number of attempts per problem )
# </explanation>


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

print(engine.data.users.keys())

# Saving the engine
# engine.save()
