import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


def load_dataset(file_path):

    return pd.read_csv(file_path)


def profile_data(data):
    profile = {
        'missing_values': data.isnull().sum().sum(),
        'data_types': data.dtypes.to_dict(),
        'summary_stats': data.describe().to_dict()
    }
    return profile


def train_model(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def compare_models(model_results):
    best_model = max(model_results, key=lambda x: x['accuracy'])
    return best_model


class TestDataPulse(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, size=100)
        })

    def test_load_dataset(self):
        data = load_dataset('dummy.csv')
        self.assertIsInstance(data, pd.DataFrame,
                              "Dataset should be a pandas DataFrame")

    def test_profile_data(self):
        profile = profile_data(self.data)
        self.assertIn('missing_values', profile,
                      "Profile should include missing values")
        self.assertIn('data_types', profile,
                      "Profile should include data types")
        self.assertIn('summary_stats', profile,
                      "Profile should include summary statistics")

    def test_train_model(self):
        model, accuracy = train_model(self.data, 'target')
        self.assertIsInstance(model, LogisticRegression,
                              "Model should be a LogisticRegression instance")
        self.assertIsInstance(
            accuracy, float, "Accuracy should be a float value")

    def test_compare_models(self):
        model_results = [
            {'model': 'Model1', 'accuracy': 0.8},
            {'model': 'Model2', 'accuracy': 0.85},
            {'model': 'Model3', 'accuracy': 0.75}
        ]
        best_model = compare_models(model_results)
        self.assertEqual(best_model['model'], 'Model2',
                         "Best model should be Model2 with the highest accuracy")


if __name__ == '__main__':
    unittest.main()
