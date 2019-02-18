from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
import time
import pandas as pd

regression_algos = [LinearRegression, RandomForestRegressor, ExtraTreesRegressor]
classification_algos = [RandomForestClassifier, ExtraTreesClassifier, LogisticRegression]

def compare_algos(x_train, x_test, y_train, y_test, pp_accuracies = False):

    algos_to_be_compared = get_algos_to_be_compared(y_train)

    accuracies = []
    index = 0
    total_time = 0

    for algo in algos_to_be_compared:
        start_time = time.time()
        model = algo()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        end_time = time.time()
        time_taken = end_time - start_time
        total_time += time_taken

        print("Completed executing " + algos_to_be_compared[index].__name__ +" algorithm")
        print("\tTime Taken : " + str(time_taken) + " seconds")

        index+=1

    if pp_accuracies:
        pp_accuracies_helper(accuracies, algos_to_be_compared)

    best_accuracy_index = accuracies.index(max(accuracies))

    print("Total Time Taken : " + str(total_time) + " seconds")

    return algos_to_be_compared[best_accuracy_index].__name__


def get_distribution(y_train):
    unique_values = len(y_train.unique())
    total_values = len(y_train)
    distribution = (unique_values / total_values) * 100

    return distribution

def get_algos_to_be_compared(y_train):
    distribution = get_distribution(y_train)
    algos_to_be_compared = []

    if distribution < 10:
        print("Classification Algorithms are being run")
        algos_to_be_compared = classification_algos
    else:
        print("Regression Algorithms are being run")
        algos_to_be_compared = regression_algos

    return algos_to_be_compared

def pp_accuracies_helper(accuracies, algos_to_be_compared):
    d = {}
    for i in range(len(accuracies)):
        d[algos_to_be_compared[i].__name__] = accuracies[i]

    d = dict(sorted(d.items(), key = lambda x : x[1]))

    for i in d:
        print("" + i + "\t" + "{0:.4f}".format(round(d[i],4)))


def help_algos():
    return "I will help"