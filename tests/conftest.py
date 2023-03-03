import pytest
import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestClassifier


class ClassifierDataTest(object):
    def __init__(self):
        self._input_df = pd.DataFrame(data={'age': [3.5, 22, 3.5, 30, 4],
                                            'Pclass': [1, 2, 3, 1, 2],
                                            'women_child': [2, 2, 2, 0, 2],
                                            'female': [13, 13, 0, 0, 0],
                                            'survived': [1, 0, 0, 1, 0]})
        self.features = ['age', 'Pclass', 'women_child', 'female']
        self.target_col = 'survived'

        self._prediction_data = pd.DataFrame(data={
            'PassengerId': ["pass_1", "pass_10", "pass_100", "pass_502", "pass_1234"],
            'age': [5, 42, 75, 30, 2],
            'Pclass': [3, 2, 1, 1, 3],
            'women_child': [2, 0, 0, 2, 2],
            'female': [13, 0, 0, 13, 0]})

    # @pytest.fixture(scope='session')
    def training_data(self):
        return self._input_df

    # @pytest.fixture(scope='session')
    def prediction_data(self):
        return self._prediction_data

    # @pytest.fixture(scope='session')
    def rf_cls_model(self):
        feats = self._input_df[self.features].values
        target = self._input_df[self.target_col].values
        rf_model = RandomForestClassifier()
        rf_model.fit(X=feats, y=target)
        return rf_model

    # @pytest.fixture(scope='function')
    def get_data_rf_classification_model(self):
        # train_data, feature_cols, target_col, n_split
        return [self._input_df, self.features, self.target_col]


@pytest.fixture(scope='module')
def get_rf_classification_model_training_data():
    return ClassifierDataTest().get_data_rf_classification_model()


@pytest.fixture(scope='module')
def get_classifier_model():
    return ClassifierDataTest().rf_cls_model()


@pytest.fixture(scope='module')
def get_prediction_data():
    return ClassifierDataTest().prediction_data()


class StorageInterfaceDataTest:
    def __init__(self):
        pass
