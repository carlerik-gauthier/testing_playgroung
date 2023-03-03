import pandas as pd
import pytest
import logging
import pickle
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

# import tests.conftest as conftest
# import pytest_mock
# from pytest_mock import mocker
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from models import classifier as cls
from gcp_interface.storage_interface import StorageInterface
# from pathlib import Path
# from _pytest.logging import LogCaptureFixture

DATA = pd.DataFrame(data={"A": [1, 2, 3], "B": ["a", "b", "c"]})

import logging
import warnings
warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()


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

    for att in mock_model.__dict__.keys():
        # TODO : check from here
        logger.info(f"mock : {getattr(mock_model, att)}")
        logger.info(f" model : {getattr(model, att)}")
        assert getattr(mock_model, att) == getattr(model, att)


@pytest.mark.skip("not yet")
def test_get_titanic_survival_prediction(ClassifierDataTest, caplog, mocker):
    # mock preprocess and use a fitted model
    application_data = ClassifierDataTest.prediction_data()
    mock_model = ClassifierDataTest.rf_cls_model()
    caplog.set_level(logging.INFO)

    mocker.patch('src.preprocessing.titanic_preprocessing.preprocess',
                 return_value=application_data)
    predictions_df = cls.get_titanic_survival_prediction(model=mock_model,
                                                         application_data=application_data,
                                                         data_configuration={'passenger_id': 'PassengerId'})
    records = caplog.records
    assert len(records) == 1
    assert records[0].message == f" [MODEL PREDICT] Feature columns to be used are : {ClassifierDataTest.features}"
    assert isinstance(predictions_df, pd.DataFrame)
    assert len(set(predictions_df.columns).difference({"PassengerId",
                                                       "predicted_survival_probability",
                                                       "predicted_survival"}
                                                      )
               ) == 0
    for prob, pred in predictions_df["predicted_survival_probability", "predicted_survival"].values:
        assert (pytest.approx(prob) == 0 or prob > 0) and (pytest.approx(prob) == 1 or prob < 1)
        assert pytest.approx(pred) == 0 or pytest.approx(pred) == 1


@pytest.mark.skip("not yet")
def test_train_model_in_local():
    # Testing how data is moved through the process ==> testing logger infos
    # mock preprocess, get_rf_classification_model, gs_interface.local_to_storage, get_data_from_storage
    pass


@pytest.mark.skip("not yet")
def test_predict_in_local():
    # Test skipped because predict_in_local is a sequence of calls to other functions or methods requiring interaction
    # with GCS and a trained model
    pass
