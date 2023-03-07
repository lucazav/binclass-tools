# binclass-tools: Binary Classification Tools for Python At Your Fingertips

<img src="/resources/images/logo.png" width="600" height="200" />

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/binclass-tools)
[![GitHub license](https://img.shields.io/github/license/lucazav/binclass-tools)](https://github.com/lucazav/binclass-tools/blob/main/LICENSE)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/lucazav/binclass-tools?color=orange)
[![Downloads](https://static.pepy.tech/personalized-badge/binclass-tools?period=total&units=international_system&left_color=grey&right_color=magenta&left_text=Downloads)](https://pepy.tech/project/binclass-tools)

A set of Python wrappers and interactive plots that facilitate the analysis of binary classification problems.

---

The __binclass-tools__ package makes the following available to you:

* Powerful interactive charts that simplify the analysis of a binary classifier's performance, including calibration and any amounts and costs associated with individual observations.

* A set of functions that return the values of metrics useful for measuring the performance of a binary classifier, for each threshold value if dependent on it.

* A set of functions to find the optimal threshold value calculated on both the most popular metrics associated with the binary classifier under analysis, and any costs associated with each of the 4 categories in the confusion matrix.

* A set of generic wrappers that help the analyst in daily operations dealing with binary classifications.

On [Towards Data Science](https://towardsdatascience.com/) you will find the following article describing the theory behind all the functions of the package and the path that led me to create a package for analyzing binary classifications that also included calculating optimal threshold values for specific metrics:

[Finding the Best Classification Threshold for Imbalanced Classifications with the Interactive Confusion Matrix and Line Charts](https://medium.com/towards-data-science/finding-the-best-classification-threshold-for-imbalanced-classifications-with-interactive-plots-7d65828dda38)

## Quick Start

### Requirements and Installation

The project is based on:
* Python 3.6+
* A set of the most popular packages used for working with data
* Plotly for interactive plots

If you do not have Python, install it first. Then, in your favorite conda or virtual environment, simply do:

```
pip install binclass-tools
```

or, if you want to install the development version directly from github:


```
pip install git+https://github.com/lucazav/binclass-tools
```

## New from version 1.0.0: 
**plotting functions' behaviour**: from version 1.0.0, the behavior of functions that generate plots has changed: Figure (Plotly) objects, dictionary-like, will be returned and not shown directly when the function is called. 

## New from version 1.1.0: 
**optimal thresholds returned in the confusion matrix plot**: from version 1.1.0 the optimal thresholds dataframe will correspond to the thresholds that give the best value of the implemented metrics (or the minimal Cost) for the given set of data.

The functions that implement the GHOST method to compute optimal thresholds are still available.

## Example Usage

Let's import both the usual libraries needed to work with the data and the binclass-tools one:

```python
import numpy as np
import pandas as pd
import bctools as bc
```

In addition, since we will train a classifier on randomly generated data via RandomForest, let's also import some useful functions for the purpose:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

Let's then train our model that we will use as a classifier to analyse thanks to the functions of _binclass-tools_:

```python
# Generate a binary imbalanced classification problem, with 80% zeros and 20% ones.
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=14, n_redundant=0,
                           random_state=12, shuffle=False, weights = [0.8, 0.2])

# Train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=0)

# Train a RF classifier
cls = RandomForestClassifier(max_depth=6, oob_score=True)
cls.fit(X_train, y_train)
```

Having trained the model, let's calculate the estimated probabilities of the predictions obtained from the training and testing datasets:

```python
# Get prediction probabilities for the train set
train_predicted_proba = cls.predict_proba(X_train)[:,1]

# Get prediction probabilities for the test set
test_predicted_proba = cls.predict_proba(X_test)[:,1] 
```
Let's generate some known graphs with the functions in the binclass-tools package to check the overall behavior of the model on the test set.
Note that it's possible to customize the main title and choose whether to display the plotly bar mode through the parameters `title` and `show_display_modebar` in every graphical function of this library.

We can start by visualizing the _Receiver Operating Characteristic (ROC) Curve_, using the following function, which also returns the value of the area under the curve:

```python
ROC_plot, area_under_ROC = bc.curve_ROC_plot(true_y = y_test, 
                                             predicted_proba = test_predicted_proba)
ROC_plot  #or ROC_plot.show(), both work 
```

Which returns the plot:

![ROC Curve for the Test Set](/resources/images/01-ROC-curve-test.png)

and returns the AUC value:

```python
>>> area_under_ROC
0.9748427672955975
```

Next, you can visualize the _Precision-Recall (PR) Curve_ plot with the iso-Fbeta curves. 
First, let's recall the definition of the F-beta score: it is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.
The beta parameter determines the weight of recall in the combined score. beta < 1 lends more weight to precision, while beta > 1 favors recall.
An iso-Fbeta curve thus contains, by definition, all points in the precision-recall space whose F-beta scores are equal. 
The function `curve_PR_plot` allows us to display ISO curves associated with F-beta score values of 0.2, 0.4, 0.6 and 0.8. The function takes as input the `beta` parameter (set to 1 as default value):

```python
PR_plot, area_under_PR = bc.curve_PR_plot(true_y = y_test, 
                                          predicted_proba = test_predicted_proba,
                                          beta = 1)
PR_plot
```

Here the plot returned:

![Precision-Recall Plot with isoF1 Curves for the Test Set](/resources/images/02-PR-isoF1-plot-test.png)

This function also returns, as in the ROC curve case, the value of the area under the curve:
```python
>>> area_under_PR 
0.9295134692043583
```

For a more in-depth analysis of the model's predicted probabilities, we can visualize through violin plots the distribution of the probabilities grouped by the relative true class and, for each threshold, see whether the predicted probability for each data point generates a correct prediction or not.
The following binclass-tools function performs the tasks just mentioned, taking as input the size of the step separating one threshold value from the other (always considering the extremes 0 and 1 inclusive):

```python
threshold_step = 0.05

violin_plot = bc.predicted_proba_violin_plot(true_y = y_test, 
                                             predicted_proba = test_predicted_proba, 
                                             threshold_step = threshold_step)
violin_plot                                           
```

Here the interactive plot returned:

![Interactive Probabilities Violin Plot for the Test Set](/resources/images/03-interactive-violin-plot-test.png)

Another useful tool to visualize the probabilities density is the `predicted_proba_density_curve_plot` function, that plots for each true class either the kernel density estimation curve (default) or the normal distribution curve, depending on the `curve_type` parameter. For each threshold, that can be selected through a slider, we can see the regions that are correctly or incorrectly classified:

```python
threshold_step = 0.05
curve_type = 'kde' #'kde' is the default value, can also be set to 'normal'

density_curve_kde = bc.predicted_proba_density_curve_plot(true_y = y_test, 
                                                          predicted_proba = test_predicted_proba, 
                                                          threshold_step = threshold_step,
                                                          curve_type = curve_type)
density_curve_kde                                                          
```

Here the returned plot:

![Interactive Probabilities Density Plot for the Test Set](/resources/images/04-interactive-density-plot-test.png)

Afterwards, we can conduct a more detailed analysis of the model's performance.
Let's set up a set of variables to pass as parameters in the subsequent binclass-tools functions we will use. 
We are going to do first an analysis of how the model performs on the training dataset, these are the variables we will need:

* The size of the step separating one threshold value from the other (always considering the extremes 0 and 1 inclusive).

* The list of individual amounts associated with each of the observables in the test dataset (since the dataset is generated by random values, the absolute value of column 13 is considered as the amount column).

* Which currency symbol to use.

* The dictionary of costs associated with each of the 4 categories of the confusion matrix. It is possible to associate a single numerical value to be considered as the average cost for each observation in that category, or a list of values to be associated with each observation. Clearly, the length of the lists in the dictionary must all be the same length, equal to the number of observations in the dataset under analysis (in our case the test dataset).

Specifically, you have this:

```python
# set params for the train dataset
threshold_step = 0.05
amounts = np.abs(X_train[:, 13])
currency = '$'

# The function get_cost_dict can be used to define the dictionary of costs.
# It takes as input, for each class, a float or a list of floats. 
# Lists must have coherent lenghts 

train_cost_dict = bc.get_cost_dict(TN = 0, FP = 10, FN = np.abs(X_train[:, 12]), TP = 0)
```

At this point we can visualize the _Interactive Confusion Matrix_ on the training dataset, including the optimal threshold for all the available metrics:

```python
cf_fig, var_metrics_df, invar_metrics_df, opt_thresh_df = bc.confusion_matrix_plot(
    true_y = y_train, 
    predicted_proba = train_predicted_proba, 
    threshold_step = threshold_step, 
    amounts = amounts, 
    cost_dict = train_cost_dict, 
    currency = currency,
    random_state = 123,
    title = 'Interactive Confusion Matrix for the Training Set')
cf_fig
```

Here the figure returned:

![Interactive Confusion Matrix for the Training Set](/resources/images/05-interactive-confusion-matrix-train.png)

As you can see, the interactive confusion matrix plot also returns metrics related dataframes that can be used in your code if needed. One is the _threshold dependent metrics dataframe_:

|    |   threshold |   accuracy |   balanced_accuracy |   cohens_kappa |   f1_score |   matthews_corr_coef |   precision |   recall |
|---:|------------:|-----------:|--------------------:|---------------:|-----------:|---------------------:|------------:|---------:|
|  0 |        0    |     0.2025 |              0.5    |         0      |     0.3368 |               0      |      0.2025 |   1      |
|  1 |        0.05 |     0.3988 |              0.623  |         0.1168 |     0.4025 |               0.249  |      0.2519 |   1      |
|  2 |        0.1  |     0.7475 |              0.8417 |         0.4664 |     0.616  |               0.5515 |      0.4451 |   1      |
|  3 |        0.15 |     0.8988 |              0.9365 |         0.7358 |     0.8    |               0.7629 |      0.6667 |   1      |
|  4 |        0.2  |     0.9462 |              0.964  |         0.8479 |     0.8822 |               0.857  |      0.7931 |   0.9938 |
|  5 |        0.25 |     0.9812 |              0.9813 |         0.9431 |     0.955  |               0.9437 |      0.9298 |   0.9815 |
|  6 |        0.3  |     0.9875 |              0.983  |         0.9615 |     0.9693 |               0.9615 |      0.9634 |   0.9753 |
|  7 |        0.35 |     0.99   |              0.9822 |         0.9689 |     0.9752 |               0.9689 |      0.9812 |   0.9691 |
|  8 |        0.4  |     0.9825 |              0.9591 |         0.9443 |     0.9551 |               0.9454 |      0.9933 |   0.9198 |
|  9 |        0.45 |     0.9712 |              0.9313 |         0.9065 |     0.9241 |               0.9098 |      0.9929 |   0.8642 |
| 10 |        0.5  |     0.9612 |              0.9043 |         0.8708 |     0.8942 |               0.8782 |      1      |   0.8086 |
| 11 |        0.55 |     0.9388 |              0.8488 |         0.7862 |     0.8218 |               0.8048 |      1      |   0.6975 |
| 12 |        0.6  |     0.91   |              0.7778 |         0.666  |     0.7143 |               0.7066 |      1      |   0.5556 |
| 13 |        0.65 |     0.8838 |              0.713  |         0.542  |     0.5974 |               0.6097 |      1      |   0.4259 |
| 14 |        0.7  |     0.8675 |              0.6728 |         0.4573 |     0.5138 |               0.5445 |      1      |   0.3457 |
| 15 |        0.75 |     0.8438 |              0.6142 |         0.3207 |     0.3719 |               0.437  |      1      |   0.2284 |
| 16 |        0.8  |     0.8238 |              0.5648 |         0.192  |     0.2295 |               0.3258 |      1      |   0.1296 |
| 17 |        0.85 |     0.805  |              0.5185 |         0.0578 |     0.0714 |               0.1725 |      1      |   0.037  |
| 18 |        0.9  |     0.8012 |              0.5093 |         0.0292 |     0.0364 |               0.1218 |      1      |   0.0185 |
| 19 |        0.95 |     0.7975 |              0.5    |         0      |     0      |               0      |      1      |   0      |
| 20 |        1    |     0.7975 |              0.5    |         0      |     0      |               0      |      1      |   0      |

The second is the _threshold invariant metrics dataframe_:

|    | invariant_metric   |   value |
|---:|:-------------------|--------:|
|  0 | roc_auc            |  0.9992 |
|  1 | pr_auc             |  0.9971 |
|  2 | brier_score        |  0.0438 |

The third and last one is a dataframe containing the _optimal threshold values_ for each implemented metric. The *optimal threshold* is the one that corresponds to the best value of the given metric (or the minimal Cost) for the **given set of data**:

|    | metric   |   optimal_threshold |
|---:|:-------------------|--------------------:|
|  0 | kappa              |                0.35  |
|  1 | mcc                |                0.35  |
|  2 | f1_score           |                0.35  |
|  3 | f2_score           |                0.25 |
|  4 | f05_score          |                0.35 |
|  5 | cost               |                0.35 |


Let's now use a different approach to compute general optimal thresholds: the **Ghost method**. 

The previous optimal threshold values refer to the thresholds that are associated with the best metric values for the given set of data.
With the GHOST method (specifically designed for imbalanced datasets) we can obtain thresholds that generally optimize the given metrics. 
We borrowed the code for calculating GHOST optimal threshold values directly from the [GHOST repository](https://github.com/rinikerlab/GHOST), introducing more metrics and optimizing the calculations using parallelism.

The `N_subset`, `subset_size`, and `with_replacement` parameters are specific to the GHOST algorithm. For more details, you can refer directly to the [paper introducing the GHOST method](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160).

```python
bc.thresholds.get_ghost_optimal_thresholds_df(
    optimize_threshold = 'all',
    threshold_values = threshold_values,
    true_y = y_train,
    predicted_proba = train_predicted_proba,
    cost_dict = train_cost_dict, 
    # GHOST parameters (these values are also the default ones) 
    N_subsets = 70,
    subsets_size = 0.2,
    with_replacement = False, 
    
    random_state = 120)
```
This function outputs:

|    | optimized_metric   |   GHOST_optimal_threshold |
|---:|:-------------------|--------------------:|
|  0 | kappa              |                0.30  |
|  1 | mcc                |                0.30  |
|  2 | f1_score           |                0.25  |
|  3 | f2_score           |                0.25 |
|  4 | f05_score          |                0.35 |
|  5 | cost               |                0.35 |


If you are interested in optimizing a single non-cost-based threshold (specifically, one of these: 'MCC', 'Kappa', 'Fscore'), you can use the following function:

```python
opt_mcc_threshold_value = bc.thresholds.get_ghost_optimal_threshold(
    y_train, 
    train_predicted_proba, 
    threshold_values,
    ThOpt_metrics = 'MCC', # default = 'Kappa'
    
    N_subsets = 70, 
    subsets_size = 0.2, 
    with_replacement = False, # defaults

    random_seed = 120)
```

Keep in mind that if you choose _'Fscore'_ as the metric to optimize, the output wll be a tuple with 3 optimal threshold values for metrics F1, F2 and F0.5 respectively.

Specifically for cost optimization (minimization), you can use the following function:

```python
opt_cost_threshold_value = bc.thresholds.get_ghost_optimal_cost(
    y_train, 
    train_predicted_proba, 
    threshold_values,
    train_cost_dict,
    
    N_subsets = 70, 
    subsets_size = 0.2, 
    with_replacement = False, # defaults

    random_seed = 120)
```

Once the GHOST optimized threshold values have been identified through the training data, the Interactive Confusion Matrix can be plotted for the test dataset:

```python

threshold_step = 0.05
amounts = np.abs(X_test[:, 13])
currency = '$'

test_cost_dict = bc.get_cost_dict(TN = 0, FP = 10, FN = np.abs(X_test[:, 12]), TP = 0)

cf_fig_test, var_metrics_df, invar_metrics_df, opt_thresh_df = bc.confusion_matrix_plot(
    true_y = y_test, 
    predicted_proba = test_predicted_proba, 
    threshold_step = threshold_step, 
    amounts = amounts, 
    cost_dict = test_cost_dict, 
    currency = currency,
    random_state = 123)

cf_fig_test
```

![Interactive Confusion Matrix for the Test Set](/resources/images/06-interactive-confusion-matrix-test.png)

Should you need to have only the returned dataframes, without generating the interactive confusion matrix plot, there are functions specifically available for this. You can get the threshold invariant metrics dataframe as follows:

```python
invar_metrics_df = bc.utilities.get_invariant_metrics_df(true_y = y_test, 
                                      predicted_proba = test_predicted_proba)
```

You can also get the threshold dependent metrics dataframe and the confusion matrix values for a specific threshold as following:

```python
conf_matrix, metrics_fixed_thresh_df = bc.utilities.get_confusion_matrix_and_metrics_df(
    true_y = y_test, 
    predicted_proba = test_predicted_proba,
    threshold = 0.3 # default = 0.5
)
```

(Keep in mind that the confusion matrix values are returned in an array, not in a dataframe.)

Finally, the dataframe of the optimal thresholds can be also obtained directly with the following code:

```python
threshold_values = np.arange(0.05, 1, 0.05)

opt_thresh_df = bc.thresholds.get_subset_optimal_thresholds_df(
    threshold_values = threshold_values,
    true_y = y_test,
    predicted_proba = test_predicted_proba,
    cost_dict = test_cost_dict)

```

You could also be also interested in visualizing the trend of possible amounts or costs associated with each category of the confusion matrix as the threshold value changes. For this purpose there is the following function that returns an _Interactive Confusion Line Chart_:

```python
cl_fig, amount_cost_df, total_amount = bc.confusion_linechart_plot(
    true_y = y_test, 
    predicted_proba = test_predicted_proba, 
    threshold_step =  threshold_step, 
    amounts = amounts, 
    cost_dict = test_cost_dict, 
    currency = currency)
cl_fig
```
Here the plot returned:

![Interactive Confusion Line Chart](/resources/images/07-interactive-confusion-line-chart.png)

You can see that there are also black "diamonds" indicating the first threshold value in which there is a swap of the amount and cost curves. The curve swapping points can also be more than one.

This function, in addition to the plot, returns two more outputs: the total amount given by the sum of all categories and the dataframe of the amounts and costs for each category as the threshold changes:

```python
print(f'total amount: {currency}{total_amount}')

amount_cost_df 
```

In addition to the result of the total amount ($374.24), here the amounts & costs dataframe:

|    |   threshold |   amount_TN |   amount_FP |   amount_FN |   amount_TP |   cost_TN |   cost_FP |   cost_FN |   cost_TP |   total_cost |
|---:|------------:|------------:|------------:|------------:|------------:|----------:|----------:|----------:|----------:|-------------:|
|  0 |        0    |      0      |   301.374   |     0       |    72.8675  |         0 |      1590 |   0       |         0 |    1590      |
|  1 |        0.05 |     48.9919 |   252.382   |     0       |    72.8675  |         0 |      1300 |   0       |         0 |    1300      |
|  2 |        0.1  |    139.883  |   161.491   |     0       |    72.8675  |         0 |       830 |   0       |         0 |     830      |
|  3 |        0.15 |    201.993  |    99.3817  |     0       |    72.8675  |         0 |       460 |   0       |         0 |     460      |
|  4 |        0.2  |    251.804  |    49.5706  |     0       |    72.8675  |         0 |       260 |   0       |         0 |     260      |
|  5 |        0.25 |    267.401  |    33.9731  |     5.73307 |    67.1344  |         0 |       160 |   3.47131 |         0 |     163.471  |
|  6 |        0.3  |    287.28   |    14.0945  |     7.87073 |    64.9967  |         0 |        70 |  10.5798  |         0 |      80.5798 |
|  7 |        0.35 |    295.033  |     6.34141 |    12.96    |    59.9075  |         0 |        20 |  15.8962  |         0 |      35.8962 |
|  8 |        0.4  |    301.374  |     0       |    15.0905  |    57.777   |         0 |         0 |  18.9167  |         0 |      18.9167 |
|  9 |        0.45 |    301.374  |     0       |    17.1228  |    55.7447  |         0 |         0 |  19.9586  |         0 |      19.9586 |
| 10 |        0.5  |    301.374  |     0       |    34.1608  |    38.7067  |         0 |         0 |  41.8435  |         0 |      41.8435 |
| 11 |        0.55 |    301.374  |     0       |    41.0564  |    31.811   |         0 |         0 |  49.1584  |         0 |      49.1584 |
| 12 |        0.6  |    301.374  |     0       |    47.5616  |    25.3058  |         0 |         0 |  54.6559  |         0 |      54.6559 |
| 13 |        0.65 |    301.374  |     0       |    58.7947  |    14.0727  |         0 |         0 |  64.8295  |         0 |      64.8295 |
| 14 |        0.7  |    301.374  |     0       |    58.7947  |    14.0727  |         0 |         0 |  64.8295  |         0 |      64.8295 |
| 15 |        0.75 |    301.374  |     0       |    66.5553  |     6.31212 |         0 |         0 |  69.3375  |         0 |      69.3375 |
| 16 |        0.8  |    301.374  |     0       |    71.3319  |     1.53555 |         0 |         0 |  75.9399  |         0 |      75.9399 |
| 17 |        0.85 |    301.374  |     0       |    71.3319  |     1.53555 |         0 |         0 |  75.9399  |         0 |      75.9399 |
| 18 |        0.9  |    301.374  |     0       |    72.8675  |     0       |         0 |         0 |  75.9666  |         0 |      75.9666 |
| 19 |        0.95 |    301.374  |     0       |    72.8675  |     0       |         0 |         0 |  75.9666  |         0 |      75.9666 |
| 20 |        1    |    301.374  |     0       |    72.8675  |     0       |         0 |         0 |  75.9666  |         0 |      75.9666 |

Just as we have already seen with the other plots, the amount and cost dataframe can be obtained directly through a specific function. In particular, you can also choose not to report amounts, for example, if you only want to analyze costs:

```python
# this function requires a list of thresholds, instead of the step, for example:
threshold_values = np.arange(0, 1, 0.05)

# example without amounts
costs_df = bc.utilities.get_amount_cost_df(
    true_y = y_test, 
    predicted_proba = test_predicted_proba,
    threshold_values = threshold_values, 
    #amounts = amounts,  
    cost_dict = test_cost_dict)
```

It may be sometimes necessary to compare the performance of what is considered a gain (e.g., amount of TP because it escaped fraud) with what is considered a loss (amount of FN of fraud escaped from the model + fixed cost per FP representing the checking to be done on transactions that are classified as fraudulent but are not). This can be done through the _Interactive Amount-Cost Line Chart_:

```python
amount_classes = ['TP', 'FP'] 
cost_classes = 'all'

ac_fig, total_cost_amount_df = bc.total_amount_cost_plot(
    true_y = y_test, 
    predicted_proba = test_predicted_proba, 
    threshold_step = threshold_step,
    amounts = amounts, 
    cost_dict = test_cost_dict,
    amount_classes = amount_classes,
    cost_classes = cost_classes,
    currency = currency)
ac_fig
```

Here the resulting plot:

![Interactive Amount-Cost Line Chart](/resources/images/08-interactive-amount-cost-line-chart.png)

As in the other cases, this function returns a dataframe with the amount and cost values, both for each category in the confusion matrix and for selected aggregates of them, associated with each threshold:

|    |   threshold |   amount_TP |   amount_FP |   amount_sum |   cost_TN |   cost_FP |   cost_FN |   cost_TP |   cost_sum |
|---:|------------:|------------:|------------:|-------------:|----------:|----------:|----------:|----------:|-----------:|
|  0 |        0    |    72.8675  |   301.374   |    374.242   |         0 |      1590 |  0        |         0 |  1590      |
|  1 |        0.05 |    72.8675  |   266.572   |    339.44    |         0 |      1380 |  0        |         0 |  1380      |
|  2 |        0.1  |    72.8675  |   152.006   |    224.874   |         0 |       770 |  0        |         0 |   770      |
|  3 |        0.15 |    72.8675  |    88.4092  |    161.277   |         0 |       430 |  0        |         0 |   430      |
|  4 |        0.2  |    72.5494  |    61.6009  |    134.15    |         0 |       290 |  0.221014 |         0 |   290.221  |
|  5 |        0.25 |    66.5301  |    31.6006  |     98.1307  |         0 |       160 |  4.472    |         0 |   164.472  |
|  6 |        0.3  |    65.3813  |    20.9625  |     86.3437  |         0 |       100 |  9.90665  |         0 |   109.907  |
|  7 |        0.35 |    60.9562  |    12.0418  |     72.998   |         0 |        30 | 18.0882   |         0 |    48.0882 |
|  8 |        0.4  |    57.8163  |     4.85876 |     62.6751  |         0 |        10 | 18.0989   |         0 |    28.0989 |
|  9 |        0.45 |    46.3113  |     0       |     46.3113  |         0 |         0 | 34.7334   |         0 |    34.7334 |
| 10 |        0.5  |    37.5392  |     0       |     37.5392  |         0 |         0 | 42.6685   |         0 |    42.6685 |
| 11 |        0.55 |    31.2279  |     0       |     31.2279  |         0 |         0 | 49.2799   |         0 |    49.2799 |
| 12 |        0.6  |    28.4496  |     0       |     28.4496  |         0 |         0 | 51.4823   |         0 |    51.4823 |
| 13 |        0.65 |    19.7851  |     0       |     19.7851  |         0 |         0 | 58.1733   |         0 |    58.1733 |
| 14 |        0.7  |     8.36888 |     0       |      8.36888 |         0 |         0 | 68.444    |         0 |    68.444  |
| 15 |        0.75 |     1.53555 |     0       |      1.53555 |         0 |         0 | 75.9399   |         0 |    75.9399 |
| 16 |        0.8  |     1.53555 |     0       |      1.53555 |         0 |         0 | 75.9399   |         0 |    75.9399 |
| 17 |        0.85 |     0       |     0       |      0       |         0 |         0 | 75.9666   |         0 |    75.9666 |
| 18 |        0.9  |     0       |     0       |      0       |         0 |         0 | 75.9666   |         0 |    75.9666 |
| 19 |        0.95 |     0       |     0       |      0       |         0 |         0 | 75.9666   |         0 |    75.9666 |
| 20 |        1    |     0       |     0       |      0       |         0 |         0 | 75.9666   |         0 |    75.9666 |

You can also directly access the previous data with the already used `get_amount_cost_df` function, excluding for example amounts to focus on costs:

```python
# this function requires a list of thresholds, instead of the step, for example:
threshold_values = np.arange(0, 1, 0.05)

# example without amounts
costs_df = bc.utilities.get_amount_cost_df(
    true_y = y_test, 
    predicted_proba = test_predicted_proba,
    threshold_values = threshold_values, 
    #amounts = amounts,  
    cost_dict = test_cost_dict)
```

Finally, there is also a function in this first release that simplifies the extraction of observations belonging to a specific category of the confusion matrix from a scored dataframe. If you want to extract, for example, all observations belonging to the TP category, this is the code you need:

```python
# for example, if we want the True Positive data points with a 0.7 threshold:
confusion_category = 'TP'

bc.get_confusion_category_observations_df(
    confusion_category = confusion_category, 
    X_data = X_test, 
    true_y = y_test, 
    predicted_proba = test_predicted_proba, 
    threshold = 0.7 # default = 0.5
)
```

#### New functions from version 1.0.0:
To further evaluate your binary classification model, a new set of functions has been introduced, generating the following:  Cumulative Gain curve, Lift curve, Response curve, Cumulative Response curve and calibration plots. 
Note that the functions for the Cumulative Gain and Lift curves take as input the **2-dimensional array** of predicted probabilities (with the probabilities associated to the negative class as well) and have a parameter that allows to specify the positive label to be considered (when not given, it will try to automatically detect it).

Here the code for the *Cumulative Gain curve* of the previously trained classification model:

```python
cumgain_plot = bc.cumulative_gain_plot(true_y = y_test, 
                                       full_predicted_proba = cls.predict_proba(X_test),
                                       pos_label = 1,
                                       )
cumgain_plot
 ```
 The function prints the information related to the label identified as positive (even if explicitly specified in the call), in this case:
 ```
Class 0 is associated with probabilities: full_predicted_proba[:, 0]
Class 1 is associated with probabilities: full_predicted_proba[:, 1]
```

and returns the plot:

![Cumulative-Gain plot](/resources/images/09-cumulative-gain-plot.png)

The cumulative gains plotshows the percentage of targets reached when considering a chosen percentage of the records with the highest predicted probability of belonging to the target class, while the baseline represents the performance of a random model. In this case, we can see that by selecting the top 23 percent of the cases according to our model, we select 80 percent of the target class.

The *lift_curve_plot* function plots the Lift curve, also called Index plot. This graph helps answer the question: when we apply the model, sort the records by their predicted probability, and select the best n deciles, how much better is it than using no model (or a random model)?
Here the code for the function:

```python
lift_curve = bc.lift_curve_plot(true_y = y_test, 
                                full_predicted_proba = cls.predict_proba(X_test),
                                pos_label = 1,
                                )
lift_curve
```

The function prints:

```
Class 0 is associated with probabilities: full_predicted_proba[:, 0]
Class 1 is associated with probabilities: full_predicted_proba[:, 1]
```

and returns the following plot:

![Lift curve plot](/resources/images/10-lift-curve-plot.png)

In this example we can see that, by selecting 44 percent of the records with the highest predicted probability, this selection contains 2.2 times the percentage of target class observations that would be obtained with a random selection. 

The *Response curve* allows to visualize the percentage of actual target class records per decile, where the first decile is associated with the 10 percent of observation with the highest predicted probability and so on. The baseline represents the percentage of target class records in the total set. 
The function that generates *response_curve_plot* takes as input the true labels, the predicted probabilties for the positive class and the number of deciles (generally called n-tiles) we want to split out dataset into:

```python
resp_curve = bc.response_curve_plot(true_y = y_test, 
                                    predicted_proba = test_predicted_proba,
                                    n_tiles = 10,
                                    )
resp_curve
```

In the following plot we can see that when selecting the decile 2 the percentage of target class records in the selection is 65%.

![Response curve plot](/resources/images/12-response-curve-plot.png)

We can also visualise the same information cumulatively through the *cumulative response plot*:

```python
cumres_plot = bc.cumulative_response_plot(true_y = y_test, 
                                          predicted_proba = test_predicted_proba,
                                          )
cumres_plot
```
![Cumulative-Response plot](/resources/images/11-cumulative-response-plot.png)

From the plot we can see that in the first 28 percent of records, ordered by predicted probabilities, 64 percent belong to the target class.

Lastly, the following two functions help to understand how well the model is calibrated. Let's recall what calibration is through an example: if a model trained to classify images as either containing or not containing a cat is presented with 10 pictures and outputs the probability of there being a cat as 0.6 (or 60%) for every image, we expect 6 cat images to be present in the set. In general terms, probabilities returned by a classification model are calibrated when a prediction of a class with confidence `p` is correct `100*p %` of the time. 

The following function plots the *calibration curve* for the model against a baseline representing a perfectly calibrated model and computes the Expected Calibration Error, taking as input the true label and the predicted probabilities of the positive class. An optional parameter *show_gaps* (True by default) allows to visualize calibration errors for each bin:

```python 
calib_curve, ece = bc.calibration_curve_plot(true_y = y_test, 
                                             predicted_proba = test_predicted_proba, 
                                             n_bins = 10,           #default
                                             strategy = 'uniform',  #default
                                             show_gaps = True,      #default
                                             ece_bins = 'fd'        #default 
                                            )

calib_curve
```

Here the returned plot:

![Calibration plot](/resources/images/13-calibration-plot.png)

And here the ECE:

```python
>>> ece 
0.13359495868308954
```

The expected calibration error can also be directly obtained with the following function:

```python
>>> bc.utilities.get_expected_calibration_error(true_y = y_test, 
                                                predicted_proba = test_predicted_proba, 
                                                bins = 'fd'           #default
                                               )
0.13359495868308954
```                                         

This last function *calibration_plot_from_models* allows you to compare the calibration of different models by taking as input: feature dataframe (X), true labels, and one or more classification models (scikit-learn consistent, must have a predict_proba method) to compare. Optionally, a list of names for the different models can be passed as input to better identify performance in the graph.
The function returns two figure objects and a list of Expected Calibration Errors (one for each model given): the first plot represents the calibration line chart with the ECE for each estimator and the second plot shows histograms with the predicted probability distribution (one for each given model).

Let's train two more estimators first:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

lr = LogisticRegression(C=1.0)
gnb = GaussianNB()

clf_list = [lr, gnb]

for clf in clf_list:
    clf.fit(X_train, y_train)
```

Now we can compare the different calibration plots and the probability distibutions:

```python
line_fig, hist_fig, ece_list = bc.calibration_plot_from_models(X = X_test, 
                                                     true_y = y_test, 
                                                     estimators = [cls, lr, gnb],
                                                     estimator_names = ["Random Forest", "Logistic", "Naive Bayes"],
                                                     n_bins = 10,           #default
                                                     strategy = 'uniform',  #default
                                                     ece_bins = 'fd'        #default
                                                    )
```

Here the plots returned:

```
line_fig.show()
hist_fig.show()
```
![Calibration plot from models](/resources/images/14-calibration-plot-from-models.png)


And the ECEs:

```python
>>> ece_list 
[0.13359495868308954, 0.05032756223563597, 0.053718608412928796]
```

You can find the complete code in the [sample notebook](/example-notebook/example_classification_model.ipynb) provided with the repository.

## Content

### Notebook:

- **example-notebook/Example_classification_model.ipynb** 
Example of how to use the binclass-tools library.

### Dependencies:
If you are interested in using _binclass-tools_ in your own code/notebooks, you'll just need these packages:
- numpy
- pandas
- scikit-learn (>=0.22.1)
- matplotlib
- plolty
- nbformat (>= 4.2.0)

## Authors
[Luca Zavarella](https://github.com/lucazav), [Greta Villa](https://github.com/GretaVilla)

## Collaborators
[Julio Cesar Cuaran Cuaran](https://github.com/JulioCesarCuaran)

## License
This package is licensed under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.