from Enums import VotingStrategy, SimilarityStrategy, VerdictTypes
from DataCollection import Collection
from Variables import Variables
from DataTesting import Testing
from DataProcessing import Preprocessing
from DataCalculation import Calculator
from Structs import User, SubmissionStats


# Engine - main part of the project. Holds the model of
# the recommendation building
class Engine:
    class Data:
        # Engine data initialization
        # path - CSV File path (should be improved to
        # include other types of data sources)
        def __init__(self):
            self.path = str()
            self.users = dict()
            self.problems = dict()

    # Engine initialization:
    # Data, Variables, Calculator, Preprocessor
    def __init__(self, path=""):
        self.data = self.Data()
        Variables.path = path
        self.data.path = path
        self.testing = Testing(self)
        self.calculator = None
        self.preprocessor = None

    # Initialization method
    # Collection - Collects data:
    # problems, users and submissions data
    # Preprocessor - preprocesses the data
    # Calculator - calculates weights and recommendation list
    def initialize(self):
        data_collection = Collection()
        data_collection.import_from_csv(self.data.path)
        self.data.users = data_collection.users
        self.data.problems = data_collection.problems
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
        print(user.recommendations)

    # Train the model and calculate recommendation list for each user
    def execute(self):
        self.preprocessor.preprocess()
        self.calculator.calculate()

    # Run mode - only train the model, no testing
    def run(self):
        self.initialize()
        self.execute()

    # Test mode - train the model and test it
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
                    self.testing.clear_aggregates()
                    self.initialize()
                    self.testing.initialize_tests()
                    self.execute()
                    self.testing.perform_test()
                    self.testing.print_means()
        else:
            Variables.similarity_strategy = SimilarityStrategy.jaccard_neighbours
            Variables.voting_strategy = VotingStrategy.weighted
            self.testing.clear_aggregates()
            self.initialize()
            self.testing.initialize_tests()
            self.execute()
            self.testing.perform_test()
            self.testing.print_means()
            print(len(self.data.users), len(self.testing.users_test))
