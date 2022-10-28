import logging
import sys
import os
import shutil

import pandas
import pandas as pd
from google.cloud import storage

logger = logging.getLogger(__name__)


class StorageInterface:
    def __init__(self, project_name=None, credentials=None):
        self._credentials = credentials
        self._project_name = project_name

        self._gs_client = self.get_gs_client()

    @property
    def gs_client(self):
        return self._gs_client

    @property
    def credentials(self):
        return self._credentials

    @property
    def project_name(self):
        return self._project_name

    @credentials.setter
    def credentials(self, new_credentials):
        self._credentials = new_credentials
        self._gs_client = self.get_gs_client()

    @project_name.setter
    def project_name(self, new_project_name: str):
        self._project_name = new_project_name
        self._gs_client = self.get_gs_client()

    @gs_client.setter
    def gs_client(self, new_gs_client):
        self._gs_client = new_gs_client

    def get_gs_client(self):
        """
        :return: Google storage client
        """
        if self.credentials is not None:
            return storage.Client(project=self.project_name,
                                  credentials=self.credentials)

        return storage.Client(project=self.project_name)

    def get_bucket(self, bucket_name: str):
        """
        :param bucket_name: name of the bucket.
        :return: google.cloud.storage.bucket.Bucket
        """
        return self.gs_client.bucket(bucket_name)

    def storage_to_local(self,
                         data_prefix: str,
                         bucket_name: str,
                         source: str,
                         destination: str
                         ):
        """
        Transfer several data from storage to local
        :param data_prefix: the blob's core name from the data to be
        transferred
        :param bucket_name: the name of the bucket
        :param source: the path to the bucket where the data are stored
        :param destination: the destination path where the data has to be moved
        :return: None
        """
        source = '' if source is None else source
        bucket = self.get_bucket(bucket_name=bucket_name)
        prefix = os.path.join(source, data_prefix)
        if source[-1] == '/':
            prefix = source + data_prefix
        for i in bucket.list_blobs(prefix=prefix):
            local_file_path = os.path.join(
                destination, os.path.basename(i.name)
            )
            i.download_to_filename(filename=local_file_path)

    def local_to_storage(self,
                         data_name: str,
                         bucket_name: str,
                         local_dir_path: str,
                         storage_dir_path: str):
        #                    timeout=60):

        """
        Transfer data from local to storage
        :param data_name: name of the data to be transferred
        :param bucket_name: the name of the bucket
        :param local_dir_path: the path in local where data tables are
        stored
        :param storage_dir_path: the path to the storage destination to
         which data is transferred
        # :param timeout: the timeout after which the connection to google
        # storage is stopped
        :return: None
        """

        bucket = self.get_bucket(bucket_name=bucket_name)
        blob = bucket.blob(blob_name=os.path.join(storage_dir_path, data_name))
        logger.info("looking for file at {}".format(
            os.path.join(local_dir_path, data_name))
        )
        logger.info(f"bucket = {bucket}")
        logger.info(f"blob = {blob}")
        blob.upload_from_filename(
            filename=os.path.join(local_dir_path, data_name)
        )
        # ,timeout=timeout)

    def check_existence(self, data: str, bucket_name: str, source: str) -> bool:
        """
        :param data: the name of the data to look in storage
        :param bucket_name: the name of the bucket
        :param source: the path to the storage destination where the data
        is stored
        :return: True/False
        """
        bucket = self.get_bucket(bucket_name=bucket_name)
        blob = bucket.blob(source + data)
        return blob.exists()

    def load_package_to_storage(self,
                                packages: dict,
                                bucket_name: str,
                                parent_path: str
                                ):
        """
        :param packages: the list of packages to load as a dictionary
        :param bucket_name: name of the bucket of the project
        :param parent_path: the path to the local project folder
        :return:
        """
        bucket = self.get_bucket(bucket_name=bucket_name)
        for package_name, uri in packages.items():

            package_uri = uri.strip().split("gs://{bucket}/".format(
                bucket=bucket.name))[1]

            blob = bucket.blob(package_uri)

            if not blob.exists():

                logger.warning(f"""
                blob {blob} does not exist on Google Storage, uploading...""")

                blob.upload_from_filename(os.path.join(parent_path, 'package',
                                                       package_name))

                logger.info(f"blob {blob} available on Google Storage")

            else:

                logger.info(f"""
                blob {blob} does exist on Google Storage, re-uploading...""")
                blob.delete()
                blob.upload_from_filename(
                    os.path.join(parent_path, 'package', package_name)
                )
                logger.info(f"blob {blob} available on Google Storage")

    def delete_in_gs(self,
                     data_name: str,
                     bucket_name: str,
                     gs_dir_path: str = None):
        """Delete the data named_ data_name in Storage."""

        if self.exist_in_gs(data_name=data_name, bucket_name=bucket_name,
                            gs_dir_path=gs_dir_path):
            bucket = self.get_bucket(bucket_name=bucket_name)
            bucket.delete_blobs(blobs=self.list_blobs(bucket_name=bucket_name,
                                                      data_name=data_name,
                                                      gs_dir_path=gs_dir_path)
                                )

    def exist_in_gs(self,
                    data_name: str,
                    bucket_name: str,
                    gs_dir_path: str = None
                    ) -> bool:
        """
        :param data_name:
        :param bucket_name:
        :param gs_dir_path:
        :return: True if data named_ data_name exist in Storage
        """
        return len(self.list_blobs(bucket_name=bucket_name,
                                   gs_dir_path=gs_dir_path,
                                   data_name=data_name)
                   ) > 0

    def list_blobs(self,
                   bucket_name: str,
                   data_name: str,
                   gs_dir_path: str = None
                   ) -> list:
        """Return the data named_ data_name in Storage as a list of
        Storage blobs.
        """
        bucket = self.get_bucket(bucket_name=bucket_name)
        if gs_dir_path is None:
            prefix = data_name
        else:
            prefix = os.path.join(gs_dir_path, data_name)
        return list(bucket.list_blobs(prefix=prefix))

    def list_blob_uris(self,
                       bucket_name: str,
                       data_name: str,
                       gs_dir_path: str = None
                       ) -> list:
        """Return the list of the uris of Storage blobs forming the data
        named_ data_name in Storage.
        """
        bucket_uri = f"gs://{bucket_name}"
        return [os.path.join(bucket_uri, blob.name) for blob in
                self.list_blobs(bucket_name=bucket_name,
                                data_name=data_name,
                                gs_dir_path=gs_dir_path)
                ]

    def storage_to_dataframe(self,
                             bucket_name: str,
                             data_name: str,
                             gs_dir_path: str = None
                             ) -> pandas.DataFrame:
        uris = self.list_blob_uris(bucket_name=bucket_name,
                                   data_name=data_name,
                                   gs_dir_path=gs_dir_path)
        logger.info(f"[STORAGE] Looking at the following uris list :\n {uris}")
        dfs = map(lambda uri: pd.read_csv(uri), uris)
        try:
            df = pd.concat(dfs, ignore_index=True).drop(columns='Unnamed: 0')
        except KeyError:
            dfs = map(lambda uri: pd.read_csv(uri), uris)
            df = pd.concat(dfs, ignore_index=True)
        except ValueError:
            logger.error("Data is NOT available in Storage")
            sys.exit(1)

        return df

    def storage_to_dataframe_via_local(self,
                                       bucket_name: str,
                                       data_name: str,
                                       gs_dir_path: str = None,
                                       delete_in_local: bool = True,
                                       local_dir_path: str = 'temporary'
                                       ) -> pandas.DataFrame:

        # storage_to_local(self, data_prefix, bucket_name, source,
        #                  destination)

        # create an empty temporary directory
        i = 0
        while os.path.exists(local_dir_path):
            i += 1
            core = local_dir_path.split('__')[0]
            local_dir_path = core + f'__{i}'

        os.mkdir(local_dir_path)

        self.storage_to_local(data_prefix=data_name,
                              bucket_name=bucket_name,
                              source=gs_dir_path,
                              destination=local_dir_path)

        dfs = map(lambda f: pd.read_csv(os.path.join(local_dir_path, f)),
                  os.listdir(local_dir_path))

        df = pd.concat(dfs, ignore_index=True)
        # delete in local the temporary folder containing the temporary files
        # if user wants to delete it
        if delete_in_local:
            shutil.rmtree(local_dir_path)

        return df

    def dataframe_to_storage(self,
                             df: pandas.DataFrame,
                             bucket_name: str,
                             data_name: str,
                             gs_dir_path: str = None
                             ) -> None:
        # gs_path = data_name if gs_dir_path is None \
        #     else os.path.join(gs_dir_path, data_name)

        # tmp_file_name = 'tmp__' + data_name
        local_dir_path = 'temporary'
        # create an empty temporary directory
        i = 0
        while os.path.exists(local_dir_path):
            i += 1
            core = local_dir_path.split('__')[0]
            local_dir_path = core + f'__{i}'

        os.mkdir(local_dir_path)

        # upload dataframe to local
        df.to_csv(os.path.join(local_dir_path, data_name), index=False)
        # files_list = [os.path.join(local_dir_path, basename)
        #               for basename in os.listdir(local_dir_path)
        #               ]

        # upload local files to storage
        self.local_to_storage(data_name=data_name,
                              bucket_name=bucket_name,
                              local_dir_path=local_dir_path,
                              storage_dir_path=gs_dir_path)

        # delete in local the temporary folder containing the temporary files
        shutil.rmtree(local_dir_path)
