# sk_algo_assist
An assistant for algorithm selection for Data Scientists using SKLearn.

A function compare the algorithms that are suitable for the dataset and get the best algorithm
    in terms of accuracy, time of execution etc.

compare_algos(df, y, split = 0.7, reg_or_class = '', metric = None)

Parameters
----------
df: Pandas Dataframe, Required
    - Dataframe with the whole dataset, splitting is done by default or as specified by you.

Y: String, Required
    - The name of the column on which prediction is being done.

split: Float, Optinal(default = 0.7)
    - Values between 0 and 1(exclusive) are only allowed.

metric: sklearn.metrics object(default=accuracy_score(classification) or mean_absolute_error(regression))
    - Metrics Allowed:
        * Regression     : mean_absolute_error, mean_squared_error, explained_variance_score
        * Classification : f1_score, precision_score, recall_score, accuracy_score
    - Eg: compare_algos(df, y, metric = f1_score)

reg_or_class: String, Optional(default = '')
    - If '' then the algorithm checks for the distribution in the y_train using the formula: distribution = len(y_train.unique())/len(y_train) * 100. If the distribution is less than 10 classification algorithms are run, else regression algorithms are run.
    - If String the allowed values are "Regression" or "Classification".

return:String
    The name of the algorithm that obatined the highest accuracy, along with the accuracy and time taken for fitting.
