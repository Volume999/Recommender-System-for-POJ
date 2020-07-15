from Engine import Engine

TOP_N = [10]
engine = Engine(path="/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions3.csv")

for N in TOP_N:
    print("N = {}".format(N))
    engine.recommendation_size = N
    engine.test()
