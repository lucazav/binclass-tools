import unittest

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

import pandas as pd
import pickle

import bctools as bc
from bctools import curve_ROC_plot

import unittest
from unittest.mock import patch

class Test_Amounts_Cost(unittest.TestCase):
    def test_curve_ROC_plot(self):
        
        threshold_step = 0.05

        # Generate a binary imbalanced classification problem, with 80% zeros and 20% ones.
        X, y = make_classification(n_samples=1000, n_features=20,
                                   n_informative=14, n_redundant=0,
                                   random_state=12, shuffle=False, weights = [0.8, 0.2])

        # Train - test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=123)

        # Train a RF classifier
        cls = RandomForestClassifier(max_depth=6, oob_score=True, random_state=123)
        cls.fit(X_train, y_train)
        
        # Get prediction probabilities for the threshold train set
        train_predicted_proba = cls.predict_proba(X_train)[:,1]
        test_predicted_proba = cls.predict_proba(X_test)[:,1] 
        
        brier_score = brier_score_loss(y_test,test_predicted_proba)
        self.assertAlmostEqual(brier_score, 0.08207331781390739, places=3)
        
        area_under_ROC = bc.curve_ROC_plot(true_y = y_test, 
                                           predicted_proba = test_predicted_proba)
        self.assertAlmostEqual(area_under_ROC, 0.9550544562049395, places=3)

        # If something breaks inside this function, it will not be called and the test will fail.
        with patch("bctools.curve_ROC_plot") as show_patch:
            curve_ROC_plot(true_y = y_test, predicted_proba = test_predicted_proba)    
            show_patch.asser_called()
    
    
        

if __name__ == '__main__':
    unittest.main()


