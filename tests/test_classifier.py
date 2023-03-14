import numpy
import pandas
import pandas as pd
import pytest
import logging
import pickle
import os
import sys

import pytest_mock

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from models import classifier as cls
from gcp_interface.storage_interface import StorageInterface
# from pathlib import Path
# from _pytest.logging import LogCaptureFixture

import logging
import warnings
warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()

DATA = pd.DataFrame(data={"A": [1, 2, 3], "B": ["a", "b", "c"]})
DATA_2 = pd.DataFrame(data={"A": [1, 2, 3],
                            "B": ["a", "b", "c"],
                            "a": [4, 0, 0],
                            "b": [0, 4, 0],
                            "women_children_first_rule_eligible": [2, 0, 2]}
                      )

# pytest.skip("not yet", allow_module_level=True)


def test_get_data_from_storage(mocker):  # (mocker: pytest_mock.mocker):
    mocker.patch('gcp_interface.storage_interface.StorageInterface.storage_to_dataframe', return_value=DATA)
    df = cls.get_data_from_storage(gs_interface=StorageInterface(project_name='project'),
                                   data_name='toto',
                                   bucket_name="bucket")
    assert isinstance(df, pd.DataFrame)
    pd.testing.assert_frame_equal(left=df, right=DATA)


@pytest.mark.usefixtures("caplog")
@pytest.mark.parametrize('n_split', [2, 5])
def test_get_rf_classification_model(n_split,
                                     get_rf_classification_model_training_data,
                                     caplog
                                     ):
    logger.info(get_rf_classification_model_training_data)
    # train_data, feature_cols, target_col = get_rf_classification_model_training_data
    caplog.set_level(logging.INFO)
    if n_split == 5:
        # 5 is the default value
        model = cls.get_rf_classification_model(train_data=get_rf_classification_model_training_data[0],
                                                feature_cols=get_rf_classification_model_training_data[1],
                                                target_col=get_rf_classification_model_training_data[2]
                                                )
    else:
        model = cls.get_rf_classification_model(train_data=get_rf_classification_model_training_data[0],
                                                feature_cols=get_rf_classification_model_training_data[1],
                                                target_col=get_rf_classification_model_training_data[2],
                                                n_splits=n_split
                                                )

    # logger check for nmb of split
    records = caplog.records
    assert len(records) == n_split
    assert records[-1].levelname == 'INFO'
    assert f"Fold {n_split}, RMSE :" in records[-1].message
    # check we get a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    # check if model is fitted
    try:
        check_is_fitted(estimator=model)
        assert True
    except NotFittedError:
        assert False


def test_retrieve_saved_model(tmp_path,
                              get_classifier_model,
                              mocker):

    tmp_path.mkdir(exist_ok=True)  # creates a Pathlib object
    # mock call to gs_interface.storage_to_local
    mock_model = get_classifier_model
    model_name = "my_rf_classifier"
    mocker.patch('gcp_interface.storage_interface.StorageInterface.storage_to_local',
                 return_value=pickle.dump(mock_model,
                                          open(os.path.join(tmp_path.absolute(), model_name + '.pickle'), 'wb')))
    model = cls.retrieve_saved_model(gs_interface=StorageInterface(project_name='project'),
                                     bucket_name='bucket',
                                     gs_dir_path='gs_dir_path',
                                     local_dir_path=tmp_path.absolute().as_posix(),
                                     model_name=model_name)
    # see Pathlib library : as_posix() changes a Pathlib object to a string
    assert isinstance(model, RandomForestClassifier)
    # check if model is fitted
    try:
        check_is_fitted(estimator=model)
        assert True
    except NotFittedError:
        assert False

    assert len(set(mock_model.__dict__.keys()).intersection(model.__dict__.keys())) ==\
           len(set(mock_model.__dict__.keys()))

    assert len(set(mock_model.__dict__.keys()).intersection(model.__dict__.keys())) == \
           len(set(model.__dict__.keys()))

    for att in mock_model.__dict__.keys():
        mock_attribute = getattr(mock_model, att)
        model_attribute = getattr(model, att)
        if type(mock_attribute) is int or type(mock_attribute) is float:
            assert mock_attribute == model_attribute
        elif isinstance(mock_attribute, numpy.ndarray):
            assert mock_attribute.shape == model_attribute.shape
        elif isinstance(mock_attribute, list):
            assert len(model_attribute) == len(mock_attribute)
        else:
            assert isinstance(model_attribute, mock_attribute.__class__)


# @pytest.mark.skip("not yet")
@pytest.mark.usefixtures("caplog")
def test_get_titanic_survival_prediction(get_prediction_data, get_classifier_model, caplog, mocker):
    # mock preprocess and use a fitted model
    application_data, features = get_prediction_data
    mock_model = get_classifier_model
    caplog.set_level(logging.INFO)

    # we have to mock all the functions in preprocess too since the code inside preprocess is run
    mocker.patch('preprocessing.titanic_preprocessing.fill_na',
                 return_value=DATA)
    mocker.patch('preprocessing.titanic_preprocessing.women_children_first_rule',
                 return_value=DATA)
    mocker.patch('preprocessing.titanic_preprocessing.dummify_categorical',
                 return_value=DATA_2)
    mocker.patch('preprocessing.titanic_preprocessing.clean_dataframe',
                 return_value=application_data)
    # As the preprocess function computes the gender values, the value of gender_col must be one column from DATA
    predictions_df = cls.get_titanic_survival_prediction(model=mock_model,
                                                         application_data=DATA,
                                                         data_configuration={'passenger_id': 'PassengerId',
                                                                             "features": {'gender_col': 'B',
                                                                                          'age_col': 'A'}})
    records = caplog.records
    assert len(records) == 1
    assert records[0].message == f" [MODEL PREDICT] Feature columns to be used are : {features}."
    assert isinstance(predictions_df, pd.DataFrame)
    assert len(set(predictions_df.columns).difference({"PassengerId",
                                                       "predicted_survival_probability",
                                                       "predicted_survival"}
                                                      )
               ) == 0
    for prob, pred in predictions_df[["predicted_survival_probability", "predicted_survival"]].values:
        assert (pytest.approx(prob) == 0 or prob > 0) and (pytest.approx(prob) == 1 or prob < 1)
        assert pytest.approx(pred) == 0 or pytest.approx(pred) == 1


@pytest.mark.usefixtures('caplog')
def test_train_model_in_local(get_rf_classification_model_training_data, tmp_path, mocker, caplog):
    # Testing how data is moved through the process ==> testing logger infos
    input_df, features, target_col = get_rf_classification_model_training_data
    caplog.set_level(logging.INFO)
    bucket_name = "bucket"
    gs_dir_path = 'gs_uri'
    model_name = 'my_model'
    local_dir_path = tmp_path.as_posix()

    tmp_path.mkdir(exist_ok=True)
    # mock preprocess, get_rf_classification_model, gs_interface.local_to_storage, get_data_from_storage

    # mocking get_data_from_storage
    mocker.patch('models.classifier.get_data_from_storage', return_value=DATA)
    # mock content ?

    # mocking
    mocker.patch('gcp_interface.storage_interface.StorageInterface.local_to_storage',
                 return_value=None)

    # clean_dataframe is the last stage from preprocess function
    mocker.patch('preprocessing.titanic_preprocessing.clean_dataframe', return_value=input_df)

    cls.train_model_in_local(gs_interface=StorageInterface(project_name='project'),
                             bucket_name=bucket_name,
                             gs_dir_path=gs_dir_path,
                             local_dir_path=local_dir_path,
                             train_name="toto",
                             data_configuration={'passenger_id': 'PassengerId',
                                                 "features": {'gender_col': 'B',
                                                              'age_col': 'A'},
                                                 'target_col': target_col},
                             model_name="my_model"
                             )
    records = caplog.records
    # train in local has 4 main logs to be tested below
    assert len(records) >= 4
    assert records[0].message == f" [MODEL TRAIN] Feature columns are : {features}. \n Target column is {target_col}"

    filename = model_name + '.pickle'
    file_local_path = os.path.join(local_dir_path, filename)
    assert records[-3].message == f" -- Dumping model to {file_local_path}"
    assert records[-2].message == f"Uploading model to Storage : {bucket_name}/{gs_dir_path}/{model_name}"""
    local_working_directory = os.path.join(os.getcwd(), local_dir_path)
    assert records[-1].message == f"Local directory is {local_working_directory}"


@pytest.mark.skip("""Test skipped so far because predict_in_local is a sequence of calls to other functions or methods
 requiring interaction with GCS and a trained model""")
def test_predict_in_local():
    pass
