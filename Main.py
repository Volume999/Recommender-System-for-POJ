import pprint
import csv
import pandas as pd
import statistics
TOP_K = 30
TOP_N = [1, 3, 5, 10, 15, 20]
TEST_SAMPLE = [100, 200, 300, 400, 500, 600]
SOLVE_REQUIREMENT = 5
users = dict()
similarity = dict()
recommendations = dict()
problems = set()
users_test = dict()
f1_agg = list()
recall_agg = list()
p_agg = list()
# users = dictionary, key - userID value = vector of problems accepted
# similarity = dictionary, key - userID value - tuples of other userID and its similarity value
# recommendations = dictionary, key - userID, value - dictionary, key - problemID, value - its weight


def initialize():
    global users,\
        similarity,\
        recommendations,\
        problems,\
        users_test,\
        f1_agg,\
        recall_agg,\
        p_agg
    users = dict()
    similarity = dict()
    recommendations = dict()
    problems = set()
    users_test = dict()


def intersection_length(lst1, lst2):
    return len([val for val in lst1 if val in lst2])


def union_length(lst1, lst2):
    return len(set(lst1 + lst2))


def build_similarity_matrix():
    global similarity
    for i in users:
        if i not in similarity.keys():
            similarity[i] = list()
        for j in users:
            if i != j:
                sim_value = intersection_length(users[i], users[j]) / union_length(users[i], users[j])
                if not set(users[j]).issubset(users[i]) and sim_value != 0:
                # if sim_value != 1:
                    similarity[i].append((j, sim_value))
                    # print("Entry: {} {} {} {} {}".format(users[i], users[j], i, j, sim_value))
                # else:
                    # print("Subset: {} {} {}".format(users[i], users[j], sim_value))
    for i in similarity:
        similarity[i].sort(key=lambda a: a[1], reverse=True)
        # pprint.pprint(similarity[i])
        similarity[i] = [similarity[i][j] for j in range(len(similarity[i])) if j < TOP_K]


def build_recommendation_matrix(n):
    global recommendations
    for i in users.keys():
        if i not in recommendations.keys():
            recommendations[i] = dict()
        for (userID, simValue) in similarity[i]:
            for prob in users[userID]:
                if prob not in users[i]:
                    # print(i, userID, prob)
                    if prob not in recommendations[i].keys():
                        recommendations[i][prob] = simValue
                    else:
                        recommendations[i][prob] += simValue
                    # print("{}: {} {}".format(i, prob, recommendations[i][prob]))
    for i in recommendations:
        temp = list(recommendations[i].items())
        temp.sort(key=lambda a: a[1], reverse=True)
        temp = [temp[i] for i in range(len(temp)) if i < n]
        # print("{}: {}".format(i, temp))
        recommendations[i] = dict()
        for (prob, count) in temp:
            recommendations[i][prob] = count


def perform_test():
    global recall_agg, f1_agg, p_agg
    precision = 0
    recall = 0
    count = 0
    for (username, problemsSolved) in users_test.items():
        if username in recommendations.keys() and len(recommendations[username].keys()) > 0:
            # print(recommendations[username], problemsSolved)
            count += 1
            truePositive = 0
            # print(recommendations[username].keys())
            falsePositive = len(recommendations[username].keys())
            falseNegative = 0
            trueNegative = 0
            # print("{} : {} - {}".format(username, recommendations[username], problemsSolved))
            for probId in problemsSolved:
                if probId in recommendations[username].keys():
                    # print(username, probId)
                    truePositive += 1
                    falsePositive -= 1
                else:
                    falseNegative += 1
            for prob in recommendations[username]:
                if prob not in problemsSolved:
                    trueNegative += 1
            # print(truePositive, falsePositive)
            # print(truePositive, len(recommendations[username].keys()))
            precision += truePositive / len(recommendations[username].keys())
            recall += truePositive / len(problemsSolved)
    # print(precision, count)
    if count != 0:
        precision = precision / count
        recall = recall / count
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = f1 = 0
    p_agg.append(precision)
    recall_agg.append(recall)
    f1_agg.append(f1)
    print("Precision = {}, Recall = {}, F1 = {}, Count = {}".format(precision, recall, f1, count))
    # for i in recommendations:
    #     print(i, len(recommendations[i]), recommendations[i])

def preparation(TEST):
    global users, users_test, problems
    df = pd.read_csv('/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions2.csv',
                     nrows=TEST, header=0, names=['id', 'user', 'status'])
    # print(df)
    for row in df.iterrows():
        ProbId = row[1][0]
        User = row[1][1]
        Status = row[1][2]
        # print(tempProbId, tempUser, tempStatus)
        if Status == 1:
            if User not in users.keys():
                users[User] = list()
            users[User].append(ProbId)
            problems.add(ProbId)
    df = pd.read_csv('/Users/citius/Desktop/Study/SeniorThesisWork/solvewaySubmissions2.csv', skiprows=TEST,
                     header=0, names=['id', 'user', 'status'])
    # print(df)
    for row in df.iterrows():
        ProbId = row[1][0]
        User = row[1][1]
        Status = row[1][2]
        # print(tempProbId, tempUser, tempStatus)
        if User in users.keys() and Status == 1: #and ProbId in problems:
            if User not in users_test.keys():
                users_test[User] = set()
            users_test[User].add(ProbId)
    # pprint.pprint(users_test)
    delete_items = list()
    for user in users.keys():
        if user in users_test.keys():
            if len(users[user]) < SOLVE_REQUIREMENT or len(users_test[user]) < SOLVE_REQUIREMENT:
                delete_items.append(user)
    for i in delete_items:
        users.pop(i, 0)
        users_test.pop(i, 0)

for test in TEST_SAMPLE:
    print("Test_sample = {}".format(test))
    for N in TOP_N:
        print("N = {}".format(N))
        initialize()
        preparation(test)
        build_similarity_matrix()
        build_recommendation_matrix(N)
        perform_test()
# print(p_agg, recall_agg, f1_agg)
print("Precision: {}\nRecall: {}\nF1: {}".format(
    statistics.mean(p_agg),
    statistics.mean(recall_agg),
    statistics.mean(f1_agg)))