#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import warnings

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from itertools import repeat
from multiprocessing import Pool

def get_optimized_thresholds_df(optimize_threshold, threshold_values, true_y, predicted_proba,
                                cost_dict = None, 
                                N_subsets = 70, subsets_size = 0.2, with_replacement = False,
                                random_state = None):
   
    """ 
    Returns a dataframe with optimal decision thresholds, for given metrics, computed with GHOST method.
    
    Parameters
    ----------
    optimize_threshold: {'all', 'ROC', 'MCC', 'Kappa', 'Fscore', 'Cost'} 
                        or list containing allowed values except 'all' 
        metrics for which thresholds will be optimized 
        'all' is equvalent to ['ROC', 'MCC', 'Kappa', 'Fscore'] if cost_dict=None, ['ROC', 'MCC', 'Kappa', 'Fscore', 'Cost'] otherwise
    threshold_values: list of three floats 
        List of decision thresholds to screen for classification
    true_y: sequence of ints
        True labels 
    predicted_proba: sequence of floats
        predicted probabilities for class 1
    random_state: int, default=None
        Controls the randomness of the bootstrapping of the samples when optimizing thresholds with GHOST method
    
    Returns
    ----------
    optimal_thresholds_df: pandas dataframe
        Dataframe containing optimal thresholds
    """
    
    threshold_names_lst = []
    threshold_array = np.array([])
    supported_metrics = ['Kappa', 'MCC', 'ROC', 'Fscore', 'Cost']

    if optimize_threshold == 'all':
        if cost_dict:
            optimize_threshold = supported_metrics
        else:
            optimize_threshold = supported_metrics[:-1]
    
    if isinstance(optimize_threshold, str):
        optimize_threshold = [optimize_threshold]
                              
    for metric_name in optimize_threshold:
        if metric_name not in supported_metrics:
            raise ValueError(f"Metric {metric_name} not supported. Supported metrics: {str(supported_metrics)}")
        if metric_name == 'Cost':
            if cost_dict is None:
                raise TypeError("To optimize threshold for cost, cost_dict argument must not be None")
                    
    for metric_name in optimize_threshold:            
        
        if metric_name == 'Fscore':
            threshold_names_lst.append('f1_score')
            threshold_names_lst.append('f2_score')
            threshold_names_lst.append('f05_score')
        else:
            threshold_names_lst.append(metric_name.lower())
        
        if metric_name == 'Cost':
            threshold_array = np.append(threshold_array, 
                                        np.round(get_cost_optimal_threshold(true_y, predicted_proba,
                                                                          threshold_values, cost_dict, 
                                                                          N_subsets = N_subsets, subsets_size = subsets_size, 
                                                                          with_replacement = with_replacement, 
                                                                          random_seed = random_state),
                                                5))
        else:
            threshold_array = np.append(threshold_array, 
                                        np.round(get_optimal_threshold(true_y, predicted_proba,
                                                                     threshold_values, ThOpt_metrics = metric_name, 
                                                                     N_subsets = N_subsets, subsets_size = subsets_size,
                                                                     with_replacement = with_replacement, 
                                                                     random_seed = random_state),
                                                 5))
        
    threshold_array = np.ravel(threshold_array)
    optimal_thresholds_df = pd.DataFrame(zip(threshold_names_lst, threshold_array), columns = ['optimized_metric', 'optimal_threshold']) 
    return optimal_thresholds_df

def get_optimal_threshold(labels, probs, thresholds, 
                          ThOpt_metrics = 'Kappa', N_subsets = 70, 
                          subsets_size = 0.2, with_replacement = False, random_seed = None):

    """ Optimize the decision threshold based on subsets of the given set (GHOST method).
    The threshold that maximizes the chosen metric on the subsets is chosen as optimal.
    
    Parameters
    ----------
    labels: sequence of ints
        True labels
    probs: sequence of floats
        predicted probabilities for class 1
        (e.g. output from cls.predict_proba(data)[:,1])
    thresholds: list of floats
        List of decision thresholds to screen for classification
    ThOpt_metrics: str {'ROC', 'MCC', 'Kappa', 'Fscore'}, default='Kappa'
        metric for which thresholds will be optimized 
    N_subsets: int, default=70
        Number of subsets used in the optimization process
    subsets_size: float or int, default=0.2
        Size of the subsets used in the optimization process. 
        If float, represents the proportion of the dataset to include in the subsets. 
        If integer, it represents the actual number of instances to include in the subsets. 
    with_replacement: bool, default=False
        If True, the subsets are drawn randomly with replacement, without otherwise.
    random_seed: int,  default=None
        Controls the randomness of the bootstrapping of the samples 
    
    Returns
    ----------
    if ThOpt_metrics == Fscore,
    opt_thresh_f1, opt_thresh_f2, opt_thresh_fpoint5: floats
        Optimal decision thresholds for f-beta scores (beta=1, beta=2, beta=0.5)
        
    otherwise:
    opt_thresh: float
        Optimal decision threshold 
    """
    
    supported_metrics = ['Kappa', 'MCC', 'ROC', 'Fscore']
    
    if ThOpt_metrics not in supported_metrics:
        raise ValueError(f"Metric {ThOpt_metrics} not supported. Supported metrics: {str(supported_metrics)}")
            
    # seeding
    np.random.seed(random_seed)
    random_seeds = np.random.randint(N_subsets*10, size=N_subsets)  
    
    df_preds = pd.DataFrame({'labels':labels,'probs':probs})
    thresh_names = [str(x) for x in thresholds]
    for thresh in thresholds:
        df_preds = pd.concat([df_preds, pd.Series([1 if x>=thresh else 0 for x in probs], name=str(thresh))], axis=1)
        
    n = max(os.cpu_count()-1, 1)
            
    if ThOpt_metrics == 'ROC':
        sensitivity_accum = []
        specificity_accum = []
        pool = Pool(n)
        
        # Calculate sensitivity and specificity for a range of thresholds and N_subsets
        for i in range(N_subsets):
            if with_replacement:
                if isinstance(subsets_size, float):
                    Nsamples = int(df_preds.shape[0]*subsets_size)
                elif isinstance(subsets_size, int):
                    Nsamples = subsets_size                    
                df_subset = resample(df_preds, n_samples = Nsamples, stratify=labels, random_state = random_seeds[i])
                labels_subset = list(df_subset['labels'])
            else:
                df_tmp, df_subset, labels_tmp, labels_subset = train_test_split(df_preds, labels, test_size = subsets_size, 
                                                                                stratify = labels, random_state = random_seeds[i])

            result = pool.starmap(_compute_sensitivity_specificity, 
                                  zip(repeat(labels_subset), [list(df_subset[threshold]) for threshold in thresh_names]))
            result_array = np.array(result)
            sensitivity_accum.append(result_array[:, 0])
            specificity_accum.append(result_array[:, 1])
        pool.close()
    
        # determine the threshold that provides the best results on the training subsets
        median_sensitivity, std_sensitivity = _helper_calc_median_std(sensitivity_accum)
        median_specificity, std_specificity = _helper_calc_median_std(specificity_accum)
        roc_dist_01corner = (2*median_sensitivity*median_specificity)/(median_sensitivity+median_specificity)
        opt_thresh = thresholds[np.argmax(roc_dist_01corner)]
        
        return opt_thresh
        
    elif ThOpt_metrics == 'Fscore':
        recall_accum = []
        precision_accum = []
        pool = Pool(n)
        
        # Calculate sensitivity and specificity for a range of thresholds and N_subsets
        for i in range(N_subsets):
            if with_replacement:
                if isinstance(subsets_size, float):
                    Nsamples = int(df_preds.shape[0]*subsets_size)
                elif isinstance(subsets_size, int):
                    Nsamples = subsets_size                    
                df_subset = resample(df_preds, n_samples = Nsamples, stratify=labels, random_state = random_seeds[i])
                labels_subset = list(df_subset['labels'])
            else:
                df_tmp, df_subset, labels_tmp, labels_subset = train_test_split(df_preds, labels, test_size = subsets_size, 
                                                                                stratify = labels, random_state = random_seeds[i])

            result = pool.starmap(_compute_precision_recall, 
                                  zip(repeat(labels_subset), [list(df_subset[threshold]) for threshold in thresh_names]))
            result_array = np.array(result)
            precision_accum.append(result_array[:, 0])
            recall_accum.append(result_array[:, 1])
        pool.close()
    
        # determine the threshold that provides the best results on the training subsets
        median_precision, std_precision = _helper_calc_median_std(precision_accum)
        median_recall, std_recall = _helper_calc_median_std(recall_accum)
        f1 = (2*median_precision*median_recall)/(median_precision+median_recall)
        opt_thresh_f1 = thresholds[np.argmax(f1)]
        f2 = (5*median_precision*median_recall)/(4*median_precision+median_recall) 
        opt_thresh_f2 = thresholds[np.argmax(f2)]
        fpoint5 = (1.25*median_precision*median_recall)/(0.25*median_precision+median_recall)
        opt_thresh_fpoint5 = thresholds[np.argmax(fpoint5)]
        
        return opt_thresh_f1, opt_thresh_f2, opt_thresh_fpoint5
        
    else:
        score_accum = []
        pool = Pool(n)
        for i in range(N_subsets):
            if with_replacement:
                if isinstance(subsets_size, float):
                    Nsamples = int(df_preds.shape[0]*subsets_size)
                elif isinstance(subsets_size, int):
                    Nsamples = subsets_size                    
                df_subset = resample(df_preds, replace=True, n_samples = Nsamples, stratify=labels, random_state = random_seeds[i])
                labels_subset = df_subset['labels']
            else:
                df_tmp, df_subset, labels_tmp, labels_subset = train_test_split(df_preds, labels, test_size = subsets_size, 
                                                                                stratify = labels, random_state = random_seeds[i])
            result = pool.starmap(_get_metric_function(ThOpt_metrics), 
                                  zip(repeat(labels_subset), [list(df_subset[threshold]) for threshold in thresh_names]))
            score_accum.append(result)
        pool.close()

        # determine the threshold that provides the best results on the training subsets
        y_values_median, y_values_std = _helper_calc_median_std(score_accum)
        opt_thresh = thresholds[np.argmax(y_values_median)]
        
        return opt_thresh

def get_cost_optimal_threshold(labels, probs, thresholds, cost_dict, 
                               N_subsets = 70, subsets_size = 0.2, 
                               with_replacement = False, random_seed = None):

    """ Optimize the decision threshold for minimal cost based on subsets of the given set (GHOST method).
    
    Parameters
    ----------
    labels: sequence of ints
        True labels
    probs: sequence of floats
        predicted probabilities for class 1
        (e.g. output from cls.predict_proba(data)[:,1])
    thresholds: list of floats
        List of decision thresholds to screen for classification
    cost_dict: dict
        dict containing costs associated to each class (TN, FP, FN, TP)
        with keys "TN", "FP", "FN", "TP" 
        and values that can be both lists (with coherent lenghts) and/or floats  
        (output from get_cost_dict)  
    N_subsets: int, default=70
        Number of subsets used in the optimization process
    subsets_size: float or int, default=0.2
        Size of the subsets used in the optimization process. 
        If float, represents the proportion of the dataset to include in the subsets. 
        If integer, it represents the actual number of instances to include in the subsets. 
    with_replacement: bool, default=False
        If True, the subsets are drawn randomly with replacement, without otherwise.
    random_seed: int,  default=None
        Controls the randomness of the bootstrapping of the samples 
    
    Returns
    ----------
    opt_thresh: float
        Optimal decision threshold 
    """
    
    # seeding
    np.random.seed(random_seed)
    random_seeds = np.random.randint(N_subsets*10, size=N_subsets)  
    
    df_preds_costs = pd.DataFrame({'labels':labels,'probs':probs, 
                                   'cost_TN':cost_dict['TN'],
                                   'cost_FP':cost_dict['FP'],
                                   'cost_FN':cost_dict['FN'],
                                   'cost_TP':cost_dict['TP']})
    
    thresh_names = [str(x) for x in thresholds]
    for thresh in thresholds:
        df_preds_costs = pd.concat([df_preds_costs, pd.Series([1 if x>=thresh else 0 for x in probs], name=str(thresh))], axis=1)
        
    n = max(os.cpu_count()-1, 1)

    score_accum = []
    pool = Pool(n)
    for i in range(N_subsets):
        if with_replacement:
            if isinstance(subsets_size, float):
                Nsamples = int(df_preds.shape[0]*subsets_size)
            elif isinstance(subsets_size, int):
                Nsamples = subsets_size                    
            df_subset = resample(df_preds_costs, replace=True, n_samples = Nsamples, stratify=labels, random_state = random_seeds[i])
            labels_subset = df_subset['labels']
        else:
            df_tmp, df_subset, labels_tmp, labels_subset = train_test_split(df_preds_costs, labels, test_size = subsets_size, 
                                                                            stratify = labels, random_state = random_seeds[i])
            
        result = pool.starmap(_get_total_cost, 
                              zip(repeat(labels_subset), 
                                  [df_subset[[threshold, 'cost_TN', 'cost_FP', 'cost_FN', 'cost_TP']] for threshold in thresh_names]))
        score_accum.append(result)
    pool.close()

    # determine the threshold that provides the best results on the training subsets
    y_values_median, y_values_std = _helper_calc_median_std(score_accum)
    opt_thresh = thresholds[np.argmin(y_values_median)]

    return opt_thresh

def _get_metric_function(metric_name):  
    # Returns the scikit function relative to the metric_name
    if metric_name == 'Kappa':
        return metrics.cohen_kappa_score
    elif metric_name == 'MCC': 
        return _MCC_wrapper
    
def _MCC_wrapper(labels, preds):
    # Wraps scikit matthews_corrcoef function suppressing zerodivision warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return metrics.matthews_corrcoef(labels, preds)

def _precision_score_wrapper(labels, preds):
    # Wraps scikit precision_score function with parameter zero_division = 1
    return metrics.precision_score(labels, preds, zero_division = 1)

def _recall_score_wrapper(labels, preds, pos_label=1):
    # Wraps scikit recall_score function with parameter zero_division = 1
    return metrics.recall_score(labels, preds, pos_label=pos_label, zero_division = 1)

def _compute_sensitivity_specificity(labels, preds):
    # Computes sensitivity (recall) and specificity through wrapped scikit functions
    return [_recall_score_wrapper(labels, preds), 
            _recall_score_wrapper(labels, preds, pos_label=0)]

def _get_total_cost(true_y, prediction_data_df):
    # Computes total cost
    y_pred = prediction_data_df.iloc[:,0]
    cost_TN = sum(prediction_data_df[(true_y == 0) & (y_pred == 0)]['cost_TN'])
    cost_FP = sum(prediction_data_df[(true_y == 0) & (y_pred == 1)]['cost_FP'])
    cost_FN = sum(prediction_data_df[(true_y == 1) & (y_pred == 0)]['cost_FN'])
    cost_TP = sum(prediction_data_df[(true_y == 1) & (y_pred == 1)]['cost_TP'])

    return cost_TN + cost_FP + cost_FN + cost_TP

def _compute_precision_recall(labels, preds):
    # Computes precision and recall through wrapped scikit functions
    return [_precision_score_wrapper(labels, preds), 
            _recall_score_wrapper(labels, preds)]

def _helper_calc_median_std(specificity): 
    # Calculate median and std of the columns of a pandas dataframe
    arr = np.array(specificity)
    y_values_median = np.median(arr,axis=0) 
    y_values_std = np.std(arr,axis=0) 
    return y_values_median, y_values_std 

