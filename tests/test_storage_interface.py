import pytest
import os
import sys

import pytest_mock

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))
import tests.parameters_storage_interface as p_gcs

from unittest import mock
from unittest import TestCase
from gcp_interface.storage_interface import StorageInterface


import logging
import warnings
warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
@pytest.mark.parametrize("project_name, credentials", p_gcs.gs_initialization())
def test_initialization(mock_storage, project_name, credentials):
    # mock_gcs_client = mock_storage.Client(project_name=project_name, credentials=credentials).return_value
    mock_gcs_client = mock_storage.Client.return_value
    gcs = StorageInterface(project_name=project_name, credentials=credentials)
    if project_name is not None:
        assert gcs._project_name == project_name
    else:
        assert gcs._project_name is None

    if credentials is not None:
        assert gcs._credentials == credentials
    else:
        assert gcs._credentials is None
    TestCase().assertEqual(mock_gcs_client, gcs._gs_client)


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
def test_initialization_default(mock_storage):
    mock_gcs_client = mock_storage.Client.return_value
    gcs = StorageInterface()
    assert gcs._project_name is None
    assert gcs._credentials is None
    TestCase().assertEqual(mock_gcs_client, gcs._gs_client)


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
@pytest.mark.parametrize("project_name, credentials", p_gcs.credentials_property())
def test_property_credentials(mock_storage, project_name, credentials):
    _ = mock_storage.Client.return_value
    gcs = StorageInterface(project_name=project_name, credentials=credentials)
    if credentials is not None:
        assert gcs.credentials == credentials
    else:
        assert gcs.credentials is None


@pytest.mark.skip
def test_property_credentials_no_credentials():
    gcs = StorageInterface()
    assert gcs.credentials is None


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
@pytest.mark.parametrize("credentials, new_credentials", p_gcs.credentials_setter())
def test_setter_credentials(mock_storage, credentials, new_credentials):
    _ = mock_storage.Client.return_value
    gcs_1 = StorageInterface()
    gcs_2 = StorageInterface(credentials=credentials)
    gcs_3 = StorageInterface(project_name="toto", credentials=credentials)

    for gcs in [gcs_1, gcs_2, gcs_3]:
        gcs.credentials = new_credentials

        assert gcs.credentials == new_credentials


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
def test_property_gs_client(mock_storage):
    mock_gcs_client = mock_storage.Client.return_value
    gcs = StorageInterface(project_name=mock_gcs_client.project)
    TestCase().assertEqual(mock_gcs_client, gcs.gs_client)


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
def test_setter_gs_client(mock_storage):
    mock_gcs_client = mock_storage.Client.return_value

    mock_storage.Client.return_value = 'x1233444'
    mock_gcs_client_2 = mock_storage.Client.return_value

    gcs = StorageInterface(project_name=mock_gcs_client)
    gcs.gs_client = mock_gcs_client_2

    TestCase().assertNotEqual(mock_gcs_client, mock_gcs_client_2)
    TestCase().assertEqual(mock_gcs_client_2, gcs.gs_client)


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
@pytest.mark.parametrize("project_name, credentials", p_gcs.project_name_property())
def test_property_project_name(mock_storage, project_name, credentials):
    _ = mock_storage.Client.return_value
    gcs = StorageInterface(project_name=project_name, credentials=credentials)
    if project_name is not None:
        assert gcs.project_name == project_name
    else:
        assert gcs.project_name is None


@pytest.mark.skip
def test_property_project_name_no_credentials():
    gcs = StorageInterface()
    assert gcs.project_name is None


# @pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
@pytest.mark.parametrize("project_name, new_project_name", p_gcs.project_name_setter())
def test_setter_project_name(mock_storage, project_name, new_project_name):
    _ = mock_storage.Client.return_value
    gcs_1 = StorageInterface()
    gcs_2 = StorageInterface(project_name=project_name)
    gcs_3 = StorageInterface(project_name=project_name, credentials="credentials")

    for gcs in [gcs_1, gcs_2, gcs_3]:
        gcs.project_name = new_project_name

        assert gcs.project_name == new_project_name


@pytest.mark.skip(" it is a raw call to storage.Client. For other tests it simply mocked")
def test_gs_get_client():
    pass


@pytest.mark.skip
def test_gs_get_bucket():
    # https://stackoverflow.com/questions/64672497/unit-testing-mock-gcs
    pass


@pytest.mark.skip
def test_storage_to_local():
    pass


@pytest.mark.skip
def test_local_to_storage():
    pass


@pytest.mark.skip
def test_check_existence():
    pass


@pytest.mark.skip
def test_load_package_to_storage():
    pass


@pytest.mark.skip
def test_delete_in_gs():
    pass


@pytest.mark.skip
def test_list_blobs():
    pass


@pytest.mark.skip
def test_list_blob_uris():
    pass


@pytest.mark.skip
def test_storage_to_dataframe():
    pass


@pytest.mark.skip
def test_storage_to_dataframe_via_local():
    pass


@pytest.mark.skip
def test_dataframe_to_storage():
    pass
