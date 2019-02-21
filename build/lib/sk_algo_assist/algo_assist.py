from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import time
import sklearn
import pandas as pd


REGRESSION_ALGOS = [LinearRegression, RandomForestRegressor, ExtraTreesRegressor]
CLASSIFICATION_ALGOS = [RandomForestClassifier, ExtraTreesClassifier, LogisticRegression]
REGRESSION_METRICS = [mean_absolute_error, mean_squared_error, explained_variance_score]
CLASSIFICATION_METRICS = [f1_score, precision_score, recall_score, accuracy_score]
REGRESSION = "Reg"
CLASSIFICATION = "Cla"


class RegOrClassNotDefined(Exception):
    pass


class NotPandasDataFrame(Exception):
    pass


class PPAccuraciesNotDefined(Exception):
    pass


class NotPandasSeries(Exception):
    pass

class MetricNotDefined(Exception):
    pass


def compare_algos(x_train, x_test, y_train, y_test, reg_or_class = '', metric = None, pp_accuracies = False):
    '''

    A function compare the algorithms that are suitable for the dataset and get the best algorithm
    in terms of accuracy, time of execution etc.

    Parameters
    ----------
    x_train:Pandas Dataframe, Required
        The features of the training set.

    x_test:Pandas Dataframe, Required
        The features of the testing set.

    y_train:Pandas Dataframe, Required
        The label of the training set.

    y_test:Pandas Dataframe, Required
        The label of the testing set.

    pp_accuracies: Boolean, Optional(default = False)

    reg_or_class: String, Optional(default = '')
        - If '' then the algorithm checks for the distribution in the y_train using the formula: distribution = len(y_train.unique())/len(y_train) * 100. If the distribution is less than 10 classification algorithms are run, else regression algorithms are run.
        - If String the allowed values are "Regression" or "Classification".

    :return:String
        The name of the algorithm that obatined the highest accuracy, along with the accuracy and time taken for fitting.

    '''

    do_error_checking(x_train, x_test, y_train, y_test, pp_accuracies, reg_or_class, metric)

    algos_to_be_compared, algo_type = get_algos_to_be_compared(y_train, reg_or_class)

    accuracies = []
    index = 0
    total_time = 0

    for algo in algos_to_be_compared:
        start_time = time.time()
        model = algo()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = get_accuracy(y_test, y_pred, metric, algo_type)

        accuracies.append(accuracy)
        end_time = time.time()
        time_taken = end_time - start_time
        total_time += time_taken

        print("Completed executing " + algos_to_be_compared[index].__name__ +" algorithm")
        print("\tAccuracy : " + str(accuracy))
        print("\tTime Taken : " + str(time_taken) + " seconds")

        index += 1

    if pp_accuracies:
        pp_accuracies_helper(accuracies, algos_to_be_compared)

    best_accuracy_index = accuracies.index(max(accuracies))

    print("\nTotal Time Taken : " + str(total_time) + " seconds")
    print('\n\nBest Algorithm : ' + str(algos_to_be_compared[best_accuracy_index].__name__) + ' \t ' + str(accuracies[best_accuracy_index]))


def do_error_checking(x_train, x_test, y_train, y_test, pp_accuracies, reg_or_class, metric):
    if not (isinstance(x_train, pd.DataFrame) and
            isinstance(x_test, pd.DataFrame)
    ):
        raise NotPandasDataFrame("The x_train, x_test should be a Pandas Dataframe.")

    if not( isinstance(y_train, pd.Series) and
            isinstance(y_test, pd.Series)
    ):
        raise NotPandasSeries("The y_train, y_test should be a Pandas Series objects.")

    if not isinstance(pp_accuracies, bool):
        raise PPAccuraciesNotDefined("pp_accuracies should be True or False.")

    if (reg_or_class != REGRESSION and
            reg_or_class != CLASSIFICATION and
            reg_or_class != ''
    ):
        raise RegOrClassNotDefined("reg_or_class should be either 'Reg' or 'Cla' or empty.")

    if metric != None and not (callable(metric)):
        raise MetricNotDefined("Metric should be a callable from sklearn.metrics")

    if not (metric in CLASSIFICATION_METRICS or
                metric in REGRESSION_METRICS
    ):
        raise MetricNotDefined(metric.__name__ + " metric is not included")


def get_distribution(y_train):
    unique_values = len(y_train.unique())
    total_values = len(y_train)
    distribution = (unique_values / total_values) * 100

    return distribution


def get_accuracy(y_test, y_pred, metric, algo_type):
    accuracy = 0

    if metric == None:
        print("\naccuracy_score is selected as the metric\n")
        metric = accuracy_score

    if algo_type == REGRESSION:
        pass
    elif algo_type == CLASSIFICATION:
        accuracy = metric(y_test, y_pred)
    else:
        raise RegOrClassNotDefined("reg_or_class should be either 'Reg' or 'Cla' or empty.")
    return accuracy


def get_algos_to_be_compared(y_train, reg_or_class):
    algos_to_be_compared = []
    algo_type = ''

    if reg_or_class:
        if reg_or_class == REGRESSION:
            algos_to_be_compared = REGRESSION_ALGOS
            algo_type = REGRESSION
        elif reg_or_class == CLASSIFICATION:
            algos_to_be_compared = CLASSIFICATION_ALGOS
            algo_type = CLASSIFICATION
    else:
        distribution = get_distribution(y_train)

        if distribution < 10:
            print("Classification Algorithms are being run")
            algos_to_be_compared = CLASSIFICATION_ALGOS
            algo_type = CLASSIFICATION
        else:
            print("Regression Algorithms are being run")
            algos_to_be_compared = REGRESSION_ALGOS
            algo_type = REGRESSION

    return algos_to_be_compared, algo_type


def pp_accuracies_helper(accuracies, algos_to_be_compared):
    d = {}
    for i in range(len(accuracies)):
        d[algos_to_be_compared[i].__name__] = accuracies[i]

    d = dict(sorted(d.items(), key = lambda x : x[1]))

    for i in d:
        print("" + i + "\t" + "{0:.4f}".format(round(d[i],4)))