# sk_algo_assist
### An assistant for algorithm selection for Data Scientists using SKLearn.

It is often a tedious task to compare all the algorithms for a given dataset because we never know which algorithm will give us the best accuracy or minimum error.

The data scientist should run all the algorithms, compare them with the help of metrics and finally choose a algorithms that suits the dataset. This package does all this automatically and gives you the comparision between algorithm in the terms of a specific metric and also time which is another big factor for choosing a algorithm.

The package goes hand in hand with the pandas package and sklearn package algorithms and metrics classes.

Just give us the data in a pandas dataframe object along with the prediction column and sit back. 

#### We will get you the results !!!!!!!

Here is how:

Install the package using the following command:

`pip install sk_algo_assist`

### **compare_algos(df, y, split = 0.7, reg_or_class = '', metric = None)**

Parameters
----------
* df: Pandas Dataframe, Required
    - Dataframe with the whole dataset, splitting is done by default or as specified by you.

* Y: String, Required
    - The name of the column on which prediction is being done.

* split: Float, Optinal(default = 0.7)
    - Values between 0 and 1(exclusive) are only allowed.

* metric: sklearn.metrics object(default=accuracy_score(classification) or mean_absolute_error(regression))
    - Metrics Allowed:
        * Regression     : mean_absolute_error, mean_squared_error, explained_variance_score
        * Classification : f1_score, precision_score, recall_score, accuracy_score
    - Eg: compare_algos(df, y, metric = f1_score)

* reg_or_class: String, Optional(default = '')
    - If '' then the algorithm checks for the distribution in the y_train using the formula: distribution = len(y_train.unique())/len(y_train) * 100. If the distribution is less than 10           classification algorithms are run, else regression algorithms are run.
    - If String the allowed values are "Regression" or "Classification".

* return:String
    - The name of the algorithm that obatined the highest accuracy, along with the accuracy and time taken for fitting.
