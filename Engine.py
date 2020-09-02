from Enums import VotingStrategy, SimilarityStrategy
from DataCollection import Collection
from Variables import Variables
from DataTesting import Testing
from DataProcessing import Preprocessing
from DataCalculation import Calculator
from Structs import User, SubmissionStats


class Engine:
    class Data:
        def __init__(self):
            self.path = str()
            self.users = dict()
            self.problems = dict()

    def __init__(self, path=""):
        self.data = self.Data()
        Variables.path = path
        self.data.path = path
        self.testing = Testing(self)
        self.calculator = None
        self.preprocessor = None

    def initialize(self):
        data_collection = Collection()
        data_collection.import_from_csv(self.data.path)
        self.data.users = data_collection.users
        self.data.problems = data_collection.problems
        self.calculator = Calculator(self.data)
        self.preprocessor = Preprocessing(self.data)

    def execute_for_user(self, user_data):
        user = User()
        for (pid, status, count) in user_data:
            if pid in self.data.problems:
                (user.problems_solved if status == 1 else user.problems_unsolved).append(pid)
                user.submissions_stats[pid] = SubmissionStats(attempts=count)
        self.preprocessor.preprocess_user(user)
        self.calculator.calculate_user(user)
        print(user.recommendations)

    def execute(self):
        self.preprocessor.preprocess()
        self.calculator.calculate()

    def run(self):
        self.initialize()
        self.execute()

    def test(self):
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
