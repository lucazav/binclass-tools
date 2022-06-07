#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings

from sklearn import metrics

def get_cost_dict(TN = 0, FP = 0, FN = 0, TP = 0):
    
    """ 
    Creates dictionary of "confusion classes" costs
    (confusion classes are: True Negative TN, False Positive FP, False Negative FN and True Positive TP)

    Parameters
    ----------
    TN: float or sequence of floats, default=0
        cost associated to true negative predictions
    FP: float or sequence of floats, default=0
        cost associated to false positive predictions
    FN: float or sequence of floats, default=0
        cost associated to false negative predictions
    TP: float or sequence of floats, default=0
        cost associated to true positive predictions
    
    Returns
    ----------
    cost_dict: dict, default=None
        dict containing keys: "TN", "FP", "FN", "TP"
        and values corresponding to lists (with coherent lenghts) and/or floats  
    """
    confusion_class_lenghts = []
    
    if (hasattr(TN, '__iter__')) and (len(TN) == 1):
        TN = TN[0]
        
    if (hasattr(FP, '__iter__')) and (len(FP) == 1):
        FP = FP[0]
    
    if (hasattr(FN, '__iter__')) and (len(FN) == 1):
        FN = FN[0]
        
    if (hasattr(TP, '__iter__')) and (len(TP) == 1):
        TP = TP[0]
        
    for confusion_class in [TN, FP, FN, TP]:
        
        if hasattr(confusion_class, '__iter__'):
            
            confusion_class = list(confusion_class)
            confusion_class_lenghts.append(len(confusion_class))
            it = iter(confusion_class_lenghts)
            the_len = next(it)
            if not all(l == the_len for l in it):
                 raise ValueError('not all list-like confusion classes data have same length')
    
    # build cost_dict
    cost_dict = {'TN' : TN,
                 'FP' : FP,
                 'FN' : FN,
                 'TP' : TP}

    return cost_dict

def get_confusion_category_observations_df(confusion_category, X_data, true_y, predicted_proba, threshold = 0.5):
    
    """ Returns X (features) dataframe of data points related to a chosen "confusion category",
    based on given true label, predicted probabilities and decision threshold
    (confusion category is either True Negative TN, False Positive FP, False Negative FN, True Positive TP)
    
    Parameters
    ----------
    confusion_category: str {'TN', 'FP', 'FN', 'TP'}
        confusion category is either True Negative TN, False Positive FP, False Negative FN, True Positive TP
    X_data: sequence of features (nd.array or list or pandas object)
        set of features 
    true_y: sequence of ints
        True labels for X_data
    predicted_proba: sequence of floats
        predicted probabilities for the class '1' of the X_data set 
        (e.g. output from cls.predict_proba(X_data)[:,1])
    threshold: float, default=0.5
        classification threshold below which prediction label is 0, 1 otherwise
    
    Returns
    ----------
    X_filtered_df: pandas DataFrame
        X DataFrame of features for data points in chosen confusion category
    """
    
    if confusion_category not in ['TN', 'FP', 'FN', 'TP']:
        raise ValueError("confusion_class must be one of {'TN', 'FP', 'FN', 'TP'}") 

    X_df = pd.DataFrame(X_data)
    true_y_array = np.squeeze(np.array(true_y))
    predicted_proba_array = np.squeeze(np.array(predicted_proba))

    if confusion_category == 'TN':
        X_filtered_df = X_df[(true_y_array == 0) & (predicted_proba < threshold)] 

    elif confusion_category == 'FP':
        X_filtered_df =  X_df[(true_y_array == 0) & (predicted_proba >= threshold)] 

    elif confusion_category == 'FN':
        X_filtered_df = X_df[(true_y_array == 1) & (predicted_proba < threshold)] 

    else: #'TP'
        X_filtered_df = X_df[(true_y_array == 1) & (predicted_proba >= threshold)] 
        
    return X_filtered_df

def get_amount_cost_df(true_y, predicted_proba, threshold_values, amounts = None, cost_dict = None):
    
    """ 
    For each threshold, computes relative amounts and/or cost for each class (TN, FP, FN, TP)
    
    Parameters
    ----------
    true_y: sequence of ints
        True labels 
    predicted_proba: sequence of floats
        predicted probabilities for class 1
        (e.g. output from model.predict_proba(data)[:,1]) 
    threshold_values: sequence of floats 
        list of classification thresholds below which prediction label is 0, 1 otherwise
    amounts: sequence of floats, default=None
        amounts associated to each element of data 
    cost_dict: dict, default=None
        dict containing keys: "TN", "FP", "FN", "TP"
        and values corresponding to lists (with coherent lenghts) and/or floats  
        (output from get_cost_dict)
    Returns
    ----------   
    amount_cost_per_threshold_df: pandas dataframe
        Dataframe containing variables: 
        - threshold
        - if amounts is given: amounts relative to each class (TN, FP, FN, TP) 
        - if cost_dict is given: cost relative to each class (TN, FP, FN, TP) and total cost
    
    """                                                    
    if (amounts is not None) and (cost_dict is not None): #both cost and amounts 
        amount_TN = []
        amount_FP = []
        amount_FN = []
        amount_TP = []
    
        cost_TN = []
        cost_FP = []
        cost_FN = []
        cost_TP = []

        for threshold in threshold_values:
            amount_matrix = _get_amount_matrix(true_y, predicted_proba, threshold, amounts)
            amount_TN.append(amount_matrix[0,0])
            amount_FP.append(amount_matrix[0,1])
            amount_FN.append(amount_matrix[1,0])
            amount_TP.append(amount_matrix[1,1])

            cost_matrix = _get_cost_matrix(true_y, predicted_proba, threshold, cost_dict)
            cost_TN.append(cost_matrix[0,0])
            cost_FP.append(cost_matrix[0,1])
            cost_FN.append(cost_matrix[1,0])
            cost_TP.append(cost_matrix[1,1])
        
        amount_cost_per_threshold_df = pd.DataFrame(zip(threshold_values, 
                                                        amount_TN, amount_FP, amount_FN, amount_TP, 
                                                        cost_TN, cost_FP, cost_FN, cost_TP), 
                                                    columns = ['threshold', 
                                                               'amount_TN', 'amount_FP', 'amount_FN', 'amount_TP',
                                                               'cost_TN', 'cost_FP', 'cost_FN', 'cost_TP']).sort_values(by='threshold')
        
        amount_cost_per_threshold_df['total_cost'] = amount_cost_per_threshold_df[['cost_TN', 'cost_FP', 
                                                                                   'cost_FN', 'cost_TP']].apply(sum, axis = 1)        
    elif amounts is not None: # only amounts
        total_amount = sum(amounts)
        amount_TN = []
        amount_FP = []
        amount_FN = []
        amount_TP = []
        
        for threshold in threshold_values:
            amount_matrix = _get_amount_matrix(true_y, predicted_proba, threshold, amounts)
            amount_TN.append(amount_matrix[0,0])
            amount_FP.append(amount_matrix[0,1])
            amount_FN.append(amount_matrix[1,0])
            amount_TP.append(amount_matrix[1,1])
            
        amount_cost_per_threshold_df = pd.DataFrame(zip(threshold_values, 
                                                        amount_TN, amount_FP, amount_FN, amount_TP),
                                                    columns = ['threshold', 
                                                               'amount_TN', 'amount_FP', 
                                                               'amount_FN', 'amount_TP']).sort_values(by='threshold')

        
    elif cost_dict is not None: # only cost
        cost_TN = []
        cost_FP = []
        cost_FN = []
        cost_TP = []
        
        for threshold in threshold_values:
            cost_matrix = _get_cost_matrix(true_y, predicted_proba, threshold, cost_dict)
            cost_TN.append(cost_matrix[0,0])
            cost_FP.append(cost_matrix[0,1])
            cost_FN.append(cost_matrix[1,0])
            cost_TP.append(cost_matrix[1,1])
                           
        amount_cost_per_threshold_df = pd.DataFrame(zip(threshold_values, 
                                                        cost_TN, cost_FP, cost_FN, cost_TP), 
                                                    columns = ['threshold', 
                                                               'cost_TN', 'cost_FP', 'cost_FN', 'cost_TP']).sort_values(by='threshold')
        
        amount_cost_per_threshold_df['total_cost'] = amount_cost_per_threshold_df[['cost_TN', 'cost_FP', 
                                                                                   'cost_FN', 'cost_TP']].apply(sum, axis = 1)
    
    else: # no cost or amount
        raise TypeError("cost_dict and amounts can't be both None.") 
    
    return amount_cost_per_threshold_df


def get_invariant_metrics_df(true_y, predicted_proba):
   
    """ 
    Computes following metrics (based on non-thresholded predicted probabilities): 
    ROC auc macro, ROC auc weighted, Pecision-Recall auc, Brier score
    
    Parameters
    ----------
    true_y: sequence of ints 
        True labels
    predicted_proba: sequence of floats
        predicted probabilities for class 1 
        (e.g. output from model.predict_proba(data)[:,1]) 

    Returns
    ----------
    metrics_df: pandas dataframe
        Dataframe containing computed metrics
    """
    
    metrics_names = ['roc_auc', 'pr_auc', 'brier_score']
    metrics_lst = []
    metrics_lst.append(round(metrics.roc_auc_score(true_y, predicted_proba), 4))
    metrics_lst.append(round(metrics.average_precision_score(true_y, predicted_proba), 4))
    metrics_lst.append(round(metrics.brier_score_loss(true_y, predicted_proba), 4))
    
    metrics_df = pd.DataFrame(zip(metrics_names, metrics_lst), columns = ['invariant_metric', 'value']) 
    return metrics_df

def get_confusion_matrix_and_metrics_df(true_y, predicted_proba, threshold = 0.5, normalize = None):
    
    """ 
    Compute 2x2 Confusion Matrix and following metrics (based on thresholded predicted probabilities): 
    Accuracy, Balanced accuracy, F1 score, Precision, Recall, Matthews corr. coeff, Cohen's Kappa
    
    Parameters
    ----------
    true_y: sequence of ints
        True labels 
    predicted_proba: sequence of floats
        predicted probabilities for class 1
        (e.g. output from model.predict_proba(data)[:,1]) 
    threshold: float, default=0.5
        classification threshold below which prediction label is 0, 1 otherwise
    normalize: {‘true’, ‘pred’, ‘all’}, default=None
        normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. 
        If None, confusion matrix will not be normalized
        
    Returns
    ----------
    cf_matrix: 2x2 np.array
        confusion matrix    
    metrics_df: pandas dataframe
        Dataframe containing metrics
    """
    
    y_pred = [int(x >= threshold) for x in predicted_proba]
    cf_matrix = metrics.confusion_matrix(true_y, y_pred, normalize = normalize)
    
    metrics_names = ['accuracy', 'balanced_accuracy', 'f1_score', 'precision', 'recall', "cohens_kappa", 'matthews_corr_coef']
    metrics_lst = []
    metrics_lst.append(round(metrics.accuracy_score(true_y, y_pred), 4)) 
    metrics_lst.append(round(metrics.balanced_accuracy_score(true_y, y_pred), 4))
    metrics_lst.append(round(metrics.f1_score(true_y, y_pred, zero_division = 0), 4))
    metrics_lst.append(round(metrics.precision_score(true_y, y_pred, zero_division = 1), 4))
    metrics_lst.append(round(metrics.recall_score(true_y, y_pred, zero_division = 1), 4))
    metrics_lst.append(round(metrics.cohen_kappa_score(true_y, y_pred), 4))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics_lst.append(round(metrics.matthews_corrcoef(true_y, y_pred), 4))    
    
    metrics_df = pd.DataFrame(zip(metrics_names, metrics_lst), columns = ['threshold_dependent_metric', 'value']) 
        
    return cf_matrix, metrics_df

def _get_amount_matrix(true_y, predicted_proba, threshold, amounts):
    
    """ 
    Compute Amount Matrix to annotate custom plotly confusion matrix plot

    Parameters
    ----------
    true_y: sequence of ints
        True labels 
    predicted_proba: sequence of floats
        predicted probabilities for class 1
        (e.g. output from model.predict_proba(data)[:,1]) 
    threshold: float
        classification threshold below which prediction label is 0, 1 otherwise
    amounts: sequence of floats 
        amounts associated to each element of data 
        (e.g. fraud detection: amount could be the associated order amount for each order)
    
    Returns
    ----------
    amount_matrix: np.array
         matrix with amount values for each class (TN, FP, FN, TP)
    """
    prediction_data_df = pd.DataFrame(zip(true_y, predicted_proba, amounts), columns = ['true_y', 'predicted_proba', 'amounts'])

    amount_TP = sum(prediction_data_df[(prediction_data_df['true_y'] == 1) & (prediction_data_df['predicted_proba'] >= threshold)]['amounts'])
    amount_FP = sum(prediction_data_df[(prediction_data_df['true_y'] == 0) & (prediction_data_df['predicted_proba'] >= threshold)]['amounts'])
    amount_TN = sum(prediction_data_df[(prediction_data_df['true_y'] == 0) & (prediction_data_df['predicted_proba'] < threshold)]['amounts'])
    amount_FN = sum(prediction_data_df[(prediction_data_df['true_y'] == 1) & (prediction_data_df['predicted_proba'] < threshold)]['amounts'])

    amount_matrix = np.array([[amount_TN, amount_FP],
                              [amount_FN, amount_TP]])
    return amount_matrix

def _get_cost_matrix(true_y, predicted_proba, threshold, cost_dict):
    
    """ 
    Compute Cost Matrix to annotate custom plotly confusion matrix plot: 

    Parameters
    ----------
    true_y: sequence of ints
        True labels 
    predicted_proba: sequence of floats
        predicted probabilities for class 1
        (e.g. output from model.predict_proba(data)[:,1]) 
    threshold: float 
        classification threshold below which prediction label is 0, 1 otherwise
    cost_dict: dict 
        dict containing keys: "TN", "FP", "FN", "TP"
        and values corresponding to lists (with same lenght) and/or floats  
        (output from get_cost_dict) 
    Returns
    ----------
    cost_matrix: np.array
         matrix with cost values for each class (TN, FP, FN, TP)
    """
    prediction_data_df = pd.DataFrame(zip(true_y, predicted_proba), columns = ['true_y', 'predicted_proba'])
    
    for confusion_class in ['TN', 'FP', 'FN', 'TP']:
        prediction_data_df['cost_' + confusion_class] = cost_dict[confusion_class]

    cost_TN = sum(prediction_data_df[(prediction_data_df['true_y'] == 0) & (prediction_data_df['predicted_proba'] < threshold)]['cost_TN'])
    cost_FP = sum(prediction_data_df[(prediction_data_df['true_y'] == 0) & (prediction_data_df['predicted_proba'] >= threshold)]['cost_FP'])
    cost_FN = sum(prediction_data_df[(prediction_data_df['true_y'] == 1) & (prediction_data_df['predicted_proba'] < threshold)]['cost_FN'])
    cost_TP = sum(prediction_data_df[(prediction_data_df['true_y'] == 1) & (prediction_data_df['predicted_proba'] >= threshold)]['cost_TP'])

    cost_matrix = np.array([[cost_TN, cost_FP],
                            [cost_FN, cost_TP]])
    return cost_matrix
