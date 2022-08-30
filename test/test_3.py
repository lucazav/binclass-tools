import unittest
from unittest.mock import patch
from bctools import predicted_proba_violin_plot

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class Test_Amounts_Cost(unittest.TestCase):
    def test_predicted_proba_violin_plot(self):
        
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
        
        # If something breaks inside this function, it will not be called and the test will fail.
        with patch("bctools.predicted_proba_violin_plot") as show_patch:
            predicted_proba_violin_plot(true_y = y_test, 
                                        predicted_proba = test_predicted_proba)
            show_patch.asser_called()

        
        

if __name__ == '__main__':
    unittest.main()
