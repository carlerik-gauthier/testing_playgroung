import pytest
import os
import sys

import pytest_mock

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))
import tests.parameters_storage_interface as p_gcs

from unittest import mock
from unittest.mock import call
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
    gcs = StorageInterface(project_name=mock_gcs_client)
    mock_storage.Client.return_value = 'x1233444'
    mock_gcs_client_2 = mock_storage.Client.return_value
    # before setter
    TestCase().assertEqual(mock_gcs_client, gcs.gs_client)

    gcs.gs_client = mock_gcs_client_2
    # after setter
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


@pytest.mark.skip
@mock.patch("gcp_interface.storage_interface.storage")
@pytest.mark.parametrize("project_name, new_project_name", p_gcs.project_name_setter())
def test_setter_project_name(mock_storage, project_name, new_project_name):
    _ = mock_storage.Client.return_value
    gcs_1 = StorageInterface()
    gcs_2 = StorageInterface(project_name=project_name)
    gcs_3 = StorageInterface(project_name=project_name, credentials="credentials")

    def process(gcs, project_name_init, project_name_new):
        if project_name_init is not None:
            assert gcs.project_name == project_name_init
        else:
            assert gcs.project_name is None

        gcs.project_name = project_name_new

        if project_name_new is not None:
            assert gcs.project_name == project_name_new
        else:
            assert gcs.project_name is None

    for gcs, p_name in [(gcs_1, None), (gcs_2, project_name), (gcs_3, project_name)]:
        process(gcs=gcs, project_name_init=p_name, project_name_new=new_project_name)


@pytest.mark.skip(" it is a raw call to storage.Client. For other tests it is simply mocked")
def test_gs_get_client():
    pass


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
def test_gs_get_bucket(mock_storage):
    # https://stackoverflow.com/questions/64672497/unit-testing-mock-gcs
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()
    mock_bucket.name.return_value = "a-bucket-name"
    mock_gcs_client.bucket.return_value = mock_bucket
    gs = StorageInterface(project_name="project_name", credentials="credentials")
    _ = gs.get_bucket(bucket_name="a-bucket-name")
    mock_storage.Client.assert_called_once()
    mock_gcs_client.bucket.assert_called_once_with("a-bucket-name")


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
@pytest.mark.parametrize("source", ["src", None, "other/"])
def test_storage_to_local(mock_storage, source):
    data_prefix = "prefix"
    destination = "dest"

    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()

    # see https://bradmontgomery.net/blog/how-world-do-you-mock-name-attribute/
    mock_blob1 = mock.Mock()
    mock_blob2 = mock.Mock()
    name1 = mock.PropertyMock(return_value="test_1")
    name2 = mock.PropertyMock(return_value="test_2")
    type(mock_blob1).name = name1
    type(mock_blob2).name = name2
    mock_blob_list = [mock_blob1, mock_blob2]
    mock_bucket.list_blobs.return_value = mock_blob_list
    mock_gcs_client.bucket.return_value = mock_bucket

    gs = StorageInterface(project_name="project_name", credentials="credentials")
    gs.storage_to_local(data_prefix=data_prefix, bucket_name="my-bucket", source=source, destination=destination)
    source = '' if source is None else source
    prefix = os.path.join(source, data_prefix)
    if source != '' and source[-1] == '/':
        prefix = source + data_prefix

    mock_storage.Client.assert_called_once()
    # to do
    mock_gcs_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.list_blobs.assert_called_once_with(prefix=prefix)
    for blob in mock_blob_list:
        blob.download_to_filename.assert_called_once_with(filename=os.path.join(destination, blob.name))


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
def test_local_to_storage(mock_storage, caplog):
    caplog.set_level(logging.INFO)
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()
    mock_gcs_client.bucket.return_value = mock_bucket
    gs = StorageInterface(project_name="project_name", credentials="credentials")
    gs.local_to_storage(data_name="toto",
                        bucket_name="my-bucket",
                        local_dir_path="local_dir",
                        storage_dir_path="gcs_dir")
    records = caplog.records
    assert len(records) == 3
    assert records[0].message == f"looking for file at {os.path.join('local_dir', 'toto')}"
    assert records[1].message.startswith("bucket =")
    assert records[2].message.startswith("blob =")

    mock_storage.Client.assert_called_once()
    mock_gcs_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with(blob_name=os.path.join('gcs_dir', 'toto'))
    mock_bucket.blob.return_value.upload_from_filename.assert_called_once_with(
        filename=os.path.join('local_dir', 'toto'))


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
def test_check_existence(mock_storage):
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()
    mock_blob = mock.Mock()
    mock_bucket.blob.return_value = mock_blob
    mock_gcs_client.bucket.return_value = mock_bucket
    gs = StorageInterface(project_name="project_name", credentials="credentials")
    _ = gs.check_existence(bucket_name="my-bucket", data="toto", source="gcs_path/")

    mock_storage.Client.assert_called_once()
    mock_gcs_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with("gcs_path/toto")
    mock_blob.exists.assert_called_once()
    # bucket = self.get_bucket(bucket_name=bucket_name)
    # blob = bucket.blob(source + data)


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
@pytest.mark.parametrize("packages, test_nb",
                         [({"p1": "gs://bucket_name/p.bdist", "p2": " gs://bucket_name/r.sdist    "}, 0),
                          ({"p1": "gs://bucket_name/p.bdist", "p2": "gs://bucket_name/r.sdist"}, 1),
                          (dict(), 2)])
def test_load_package_to_storage(mock_storage, packages, test_nb, caplog):
    exists_return = (test_nb % 2 == 0)
    nb_calls = len(packages.keys())

    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()
    mock_blob = mock.Mock()
    bucket_name = mock.PropertyMock(return_value="bucket_name")
    type(mock_bucket).name = bucket_name
    mock_blob.exists.return_value = exists_return
    mock_bucket.blob.return_value = mock_blob
    mock_gcs_client.bucket.return_value = mock_bucket

    caplog.set_level(logging.INFO)
    gs = StorageInterface(project_name="project_name", credentials="credentials")
    _ = gs.load_package_to_storage(bucket_name="bucket_name",
                                   packages=packages,
                                   parent_path="local_path/to/project")
    records = caplog.records
    assert len(records) == 2*nb_calls
    for i in range(nb_calls):
        if exists_return:
            assert records[2 * i].levelname == "INFO"
            assert records[2 * i].message == f"""-- blob {mock_blob} does exist on Google Storage, re-uploading..."""
        else:
            assert records[2 * i].levelname == "WARNING"
            assert records[2 * i].message == f"""-- blob {mock_blob} does not exist on Google Storage, uploading..."""

        assert records[2*i+1].levelname == "INFO"
        assert records[2*i+1].message == f"blob {mock_blob} available on Google Storage"

    mock_storage.Client.assert_called_once()
    mock_gcs_client.bucket.assert_called_once_with("bucket_name")
    if nb_calls == 0:
        mock_bucket.blob.assert_not_called()
        mock_blob.exists.assert_not_called()
        mock_blob.delete.assert_not_called()
        mock_blob.upload_from_filename.assert_not_called()
    else:
        if exists_return is False:
            mock_blob.delete.assert_not_called()
        else:
            assert mock_blob.delete.call_count == nb_calls

        assert mock_bucket.blob.call_count == nb_calls
        assert mock_blob.exists.call_count == nb_calls
        assert mock_blob.upload_from_filename.call_count == nb_calls

        calls = [mock.call("p.bdist"), mock.call("r.sdist")]
        upload_calls = [call('local_path/to/project/package/p1'),
                        call('local_path/to/project/package/p2')]
        mock_bucket.blob.assert_has_calls(calls, any_order=True)  # turn last option to False to make the test fail
        # and see the list of calls
        mock_blob.upload_from_filename.assert_has_calls(upload_calls, any_order=False)


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
@pytest.mark.parametrize("existence", [True, False])
def test_delete_in_gs(mock_storage, existence, mocker):
    # is right method being called ?
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()

    mock_blob1 = mock.Mock()
    mock_blob2 = mock.Mock()
    name1 = mock.PropertyMock(return_value="test_1")
    name2 = mock.PropertyMock(return_value="test_2")
    type(mock_blob1).name = name1
    type(mock_blob2).name = name2
    mock_blob_list = [mock_blob1, mock_blob2] if existence else []
    mock_bucket.list_blobs.return_value = mock_blob_list
    mock_gcs_client.bucket.return_value = mock_bucket

    mocker.patch('gcp_interface.storage_interface.StorageInterface.exist_in_gs',
                 return_value=existence)
    gs = StorageInterface(project_name="project_name", credentials="credentials")

    _ = gs.delete_in_gs(data_name="data_name", bucket_name="bucket", gs_dir_path="gs://test")

    if existence:
        assert mock_gcs_client.bucket.call_count == 2
        # mock_gcs_client.bucket.assert_called_once_with(bucket_name="bucket")  # it will called 2x
        mock_gcs_client.bucket.assert_has_calls([call("bucket"), call("bucket")], any_order=True)
        mock_bucket.delete_blobs.assert_called_once_with(blobs=mock_blob_list)
        mock_bucket.list_blobs.assert_called_once_with(prefix="gs://test/data_name")
    else:
        mock_gcs_client.bucket.assert_not_called()
        mock_bucket.delete_blocbs.assert_not_called()


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
@pytest.mark.parametrize("existence", [True, False])
def test_exist_in_gs(mock_storage, existence):
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()

    mock_blob1 = mock.Mock()
    mock_blob2 = mock.Mock()
    mock_blob_list = [mock_blob1, mock_blob2] if existence else []
    mock_bucket.list_blobs.return_value = mock_blob_list
    mock_gcs_client.bucket.return_value = mock_bucket

    gs = StorageInterface(project_name="project_name", credentials="credentials")
    exists = gs.exist_in_gs(data_name="data_name", bucket_name="bucket", gs_dir_path="gs://test")

    mock_bucket.list_blobs.assert_called_once_with(prefix="gs://test/data_name")
    mock_gcs_client.bucket.assert_called_once_with('bucket')
    assert exists == existence


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
@pytest.mark.parametrize("gs_dir_path", ["gs://test", None])
def test_list_blobs(mock_storage, gs_dir_path):
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()
    mock_bucket.list_blobs.return_value = []
    mock_gcs_client.bucket.return_value = mock_bucket
    gs = StorageInterface(project_name="project_name", credentials="credentials")
    if gs_dir_path is not None:
        _ = gs.exist_in_gs(data_name="data_name", bucket_name="bucket", gs_dir_path=gs_dir_path)
        prefix = "gs://test/data_name"
    else:
        _ = gs.exist_in_gs(data_name="data_name", bucket_name="bucket")
        prefix = "data_name"

    mock_bucket.list_blobs.assert_called_once_with(prefix=prefix)
    mock_gcs_client.bucket.assert_called_once_with('bucket')


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
@pytest.mark.parametrize("existence", [True, False])
def test_list_blob_uris(mock_storage, existence):
    # is right method being called ?
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()

    mock_blob1 = mock.Mock()
    mock_blob2 = mock.Mock()
    name1 = mock.PropertyMock(return_value="test_1")
    name2 = mock.PropertyMock(return_value="test_2")
    type(mock_blob1).name = name1
    type(mock_blob2).name = name2
    mock_blob_list = [mock_blob1, mock_blob2] if existence else []
    mock_bucket.list_blobs.return_value = mock_blob_list
    mock_gcs_client.bucket.return_value = mock_bucket

    gs = StorageInterface(project_name="project_name", credentials="credentials")
    ll_uris = gs.list_blob_uris(data_name="data_name", bucket_name="bucket", gs_dir_path="gs://test")
    expected_output = ["gs://bucket/test_1", "gs://bucket/test_2"]
    mock_bucket.list_blobs.assert_called_once_with(prefix="gs://test/data_name")
    mock_gcs_client.bucket.assert_called_once_with('bucket')
    assert all([x == y for x, y in zip(ll_uris, expected_output)])


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
@pytest.mark.parametrize("data_df", p_gcs.gs_initialization())
def test_storage_to_dataframe(mock_storage, data_df, caplog, tmp_path, mocker):
    mock_gcs_client = mock_storage.Client.return_value
    mock_bucket = mock.Mock()
    mock_bucket.list_blobs.return_value = []
    mock_gcs_client.bucket.return_value = mock_bucket

    # is data correctly retrieved ?
    local_dir_path = tmp_path.as_posix()
    tmp_path.mkdir(exist_ok=True)

    # upload data to csv files in local_dir_path. The locations will be used in the mocker patch below
    # TBD
    #############

    caplog.set_level(logging.INFO)
    gs = StorageInterface(project_name="project_name", credentials="credentials")
    mocker.patch('gcp_interface.storage_interface.StorageInterface.list_blob_uris',
                 return_value=...)
    output_df = gs.storage_to_dataframe(bucket_name="bucket_name", data_name="data", gs_dir_path="gs://test")
    records = caplog.records
    assert len(records) > 0
    assert records[0].message.split(":")[0].strip() == "[STORAGE] Looking at the following uris list"
    # test exceptions and sys exit
    # test output
    pass

#     def storage_to_dataframe(self,
#                              bucket_name: str,
#                              data_name: str,
#                              gs_dir_path: str = None
#                              ) -> pandas.DataFrame:
#         uris = self.list_blob_uris(bucket_name=bucket_name,
#                                    data_name=data_name,
#                                    gs_dir_path=gs_dir_path)
#         logger.info(f"[STORAGE] Looking at the following uris list :\n {uris}")
#         dfs = map(lambda uri: pd.read_csv(uri), uris)
#         try:
#             df = pd.concat(dfs, ignore_index=True).drop(columns='Unnamed: 0')
#         except KeyError:
#             dfs = map(lambda uri: pd.read_csv(uri), uris)
#             df = pd.concat(dfs, ignore_index=True)
#         except ValueError:
#             logger.error("Data is NOT available in Storage")
#             sys.exit(1)
#
#         return df

@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
def test_storage_to_dataframe_via_local(mock_storage):
    # is data correctly retrieved ?
    # idem que storage_to_dataframe en tant que mock_storage
    pass


@pytest.mark.skip
@mock.patch('gcp_interface.storage_interface.storage')
def test_dataframe_to_storage(mock_storage):
    # il y aura du bucket.blob et du blob.upload_from_filename
    pass
