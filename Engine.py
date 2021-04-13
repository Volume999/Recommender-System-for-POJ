from Enums import VotingStrategy, SimilarityStrategy, VerdictTypes, RunMode
from DataCollection import get_Collection
from Variables import Variables
from DataTesting import Testing
from DataProcessing import Preprocessing
from DataCalculation import Calculator
from Structs import User, SubmissionStats
from DataSource import EnginePickle
from CustomLib import debug_timing
import pickle
import sys

sys.setrecursionlimit(50000)


# Engine - main part of the project. Holds the model of
# the recommendation building
class Engine:
    class Data:
        # Engine data initialization
        # path - CSV File path (should be improved to
        # include other types of data sources)
        def __init__(self):
            self.users = dict()
            self.problems = dict()

    # Engine initialization:
    # Data, Variables, Calculator, Preprocessor
    def __init__(self, data_source, mode):
        self.data = self.Data()
        self.testing = Testing(self)
        self.calculator = None
        self.preprocessor = None
        self.initialize(data_source)
        if mode == RunMode.test:
            self.testing.initialize_tests()
        self.execute()
        if mode == RunMode.test:
            self.test()

    # Initialization method
    # Collection - Collects data:
    # problems, users and submissions data
    # Preprocessor - preprocesses the data
    # Calculator - calculates weights and recommendation list
    def initialize(self, data_source):
        data_collection = get_Collection(data_source=data_source)
        self.data.users = data_collection.users
        self.data.problems = data_collection.problems
        # print('Users:', len(data_collection.users))
        # print('Problems:', len(data_collection.problems))
        # submissions = 0
        # for user in data_collection.users:
        #     submissions += len(data_collection.users[user].submissions_stats)
        # print('Submissions:', submissions)
        self.calculator = Calculator(self.data)
        self.preprocessor = Preprocessing(self.data)

    # Returns the recommendation list for a user
    # Input - user data - list of tuples of form
    # (Task Id, status, attempts)
    def execute_for_user(self, user_data):
        user = User()
        for (pid, status, count) in user_data:
            if pid in self.data.problems:
                (user.problems_solved if status == VerdictTypes.success.value else user.problems_unsolved).append(pid)
                user.submissions_stats[pid] = SubmissionStats(attempts=count)
        self.preprocessor.preprocess_user(user)
        self.calculator.calculate_user(user)
        return user.recommendations

    # Train the model and calculate recommendation list for each user
    def execute(self):
        self.preprocessor.preprocess()
        self.calculator.calculate()

    # Run mode - only train the model, no testing
    # def run(self):
    # self.initialize()
    # self.execute()

    # Test mode - train the model and test it
    # @debug_timing
    def test(self):
        # Full test - for each Similarity strategy
        # and for each Voting strategy perform tests
        # Else do it only for the selected strategies
        full_test = False
        if full_test:
            for similarity in SimilarityStrategy:
                for voting in VotingStrategy:
                    Variables.similarity_strategy = similarity
                    Variables.voting_strategy = voting
                    self.execute()
                    self.testing.perform_test()
                    self.testing.print_results()
        else:
            Variables.similarity_strategy = SimilarityStrategy.jaccard_neighbours
            Variables.voting_strategy = VotingStrategy.weighted
            self.testing.perform_test()
            self.testing.print_results()
            # print("Users:", len(self.data.users))
            # print("Users qualified for testing:", len(self.testing.users_test))

    def save(self):
        with open(Variables.engine_pickle_file_name_5, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


@debug_timing
def get_engine(dataSource, mode):
    if isinstance(dataSource, EnginePickle):
        with open(dataSource.file_path, 'rb') as pickle_file:
            engine = pickle.load(pickle_file)
            if mode == RunMode.test:
                # engine.testing.initialize_tests()
                engine.execute()
                engine.test()
            return engine
    else:
        return Engine(dataSource, mode)
