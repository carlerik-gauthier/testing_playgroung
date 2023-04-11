"""
The purpose of this file is to provide all connections to BQ and Storage
"""
import os
import logging
import warnings

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, BadRequest

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()

warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")


class BigqueryInterface:
    def __init__(self, project_name=None, dataset_name=None, credentials=None):
        """
        credentials (Optional[google.auth.credentials.Credentials]):
            The OAuth2 Credentials to use for this client. If not passed
            (and if no ``_http`` object is passed), falls back to the
            default inferred from the environment.
         """
        self._credentials = credentials
        self._project_name = project_name
        self._dataset_name = dataset_name

        self._bq_client = self.get_bq_client()

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def bq_client(self):
        return self._bq_client

    @property
    def credentials(self):
        return self._credentials

    @property
    def project_name(self):
        return self._project_name

    @dataset_name.setter
    def dataset_name(self, new_dataset_name: str):
        """
        :param new_dataset_name: name of the new default dataframe
        :return: None
        """
        self._dataset_name = new_dataset_name

    @credentials.setter
    def credentials(self, new_credentials):
        """
        :param new_credentials: new credentials to be used
        :return:
        """
        self._credentials = new_credentials
        self._bq_client = self.get_bq_client()

    @project_name.setter
    def project_name(self, new_project_name: str):
        """
        :param new_project_name: new project to be used as client
        :return:
        """
        self._project_name = new_project_name
        self._bq_client = self.get_bq_client()

    @bq_client.setter
    def bq_client(self, new_bq_client: bigquery.Client):
        """
        :param new_bq_client: a new Bigquery client
        :return:
        """
        self._bq_client = new_bq_client

    def get_dataset_ref(self):
        """
        :return: google.cloud.bigquery.dataset.DatasetReference
        """
        dataset_ref = bigquery.dataset.DatasetReference(
            project=self.project_name,
            dataset_id=self.dataset_name)
        return dataset_ref

    def get_table_ref(self, table_name: str):
        """
        :param table_name: name of the table
        :return: google.cloud.bigquery.table.TableReference
        """
        return self.get_dataset_ref().table(table_id=table_name)

    def query_to_bq(self,
                    query: str,
                    destination_table_name: str,
                    range_partition_by: dict = None,
                    time_partition_by: dict = None):
        """
        Runs query and loads result in BQ
        :param query: SQL Query to be executed
        :param destination_table_name: name of the dataset containing the output table to be created
        :param range_partition_by: name of the columns to use for a
        range partition. Default is None.
        REQUIRED keys are 'column_name', 'start', 'end' and 'interval'.
        :param time_partition_by: name of the columns to use for a
        range partition. Default is None. Not used if range_partition_by
        is provided.
        REQUIRED keys are 'column_name' and 'expiration_ms'.
        :return:
        """
        is_valid = self.check_query_validity(query=query)

        if not is_valid:
            raise BadRequest("""The following query has not been run since
            it is incorrect :
            {query}
            """.format(query=query))

        job_config = bigquery.QueryJobConfig()

        partition_subquery = ''
        drop_statement = ''
        if self.exists(table_name=destination_table_name):
            drop_statement = """
            DROP TABLE IF EXISTS `{destination_dataset}.{destination_table}`;
            """.format(destination_dataset=self.dataset_name,
                       destination_table=destination_table_name
                       )
        if range_partition_by is not None:
            partition_subquery = """
            {drop_statement}
            CREATE TABLE
               `{destination_dataset}.{destination_table}`
             PARTITION BY
               RANGE_BUCKET({field},
                GENERATE_ARRAY({start}, {end}, {interval}))
               AS
            """.format(drop_statement=drop_statement,
                       destination_dataset=self.dataset_name,
                       destination_table=destination_table_name,
                       field=range_partition_by['column_name'],
                       start=range_partition_by['start'],
                       end=range_partition_by['end'],
                       interval=range_partition_by['interval']
                       )

        elif time_partition_by is not None:

            partition_subquery = """
            {drop_statement}
             CREATE TABLE
               {destination_dataset}.{destination_table}
             PARTITION BY
               {date_field}
               AS
            """.format(drop_statement=drop_statement,
                       destination_dataset=self.dataset_name,
                       destination_table=destination_table_name,
                       date_field=range_partition_by['column_name']
                       )
        else:
            job_config.destination = self.get_table_ref(
                table_name=destination_table_name)
            job_config.write_disposition = \
                bigquery.WriteDisposition.WRITE_TRUNCATE

        query = partition_subquery + query
        query_job = self.bq_client.query(
            query=query,
            location='EU',
            job_config=job_config)
        query_job.result()

    def query_to_storage(self, query: str, destination_id: str, bucket_name: str):
        """
        Run query and loads result in GCS
        :param query: query to be run
        :param destination_id: Storage destination path
        :param bucket_name: Storage bucket name
        :return:
        """
        tmp_table_name = 'tmp_query_to_storage'
        tmp_table_id = '{project_name}.{dataset_name}.{tmp_table_name}'.format(
            project_name=self.project_name,
            dataset_name=self.dataset_name,
            tmp_table_name=tmp_table_name
        )
        self.query_to_bq(query=query, destination_table_name=tmp_table_name)
        self.bq_to_storage(destination_id=destination_id,
                           bucket_name=bucket_name,
                           table_name=tmp_table_name)
        self.delete_bq_table(table_id=tmp_table_id)

    def bq_to_storage(self,
                      destination_id: str,
                      table_name: str,
                      bucket_name: str):
        """
        Moves data from BQ to GS
        :param destination_id: ID of the destination file inside the
        bucket: <folder_1>/<folder_2>/ .../<name given to the file>
        :param table_name: name of the table
        :param bucket_name: name of the bucket
        :return:
        """

        destination_uri = "gs://{bucket_name}/{destination_id}".format(
                bucket_name=bucket_name, destination_id=destination_id)

        extract_job = self.bq_client.extract_table(
            source=self.get_table_ref(table_name=table_name),
            destination_uris=destination_uri,
            location='EU',
            job_config=self.get_bq_extract_job_config())
        extract_job.result()

    def storage_to_bq(self,
                      source_uris: (bigquery.client.Union[str, bigquery.client.Sequence[str]]),
                      bq_table_name: str,
                      schema=None):
        """
        :param source_uris: URIs of data files to be loaded; in format
                ``gs://<bucket_name>/<object_name_or_glob>``
        :param bq_table_name: name of the table in BQ where to load data from GCS
        :param schema: list of bigquery.SchemaField. See https://cloud.google.com/bigquery/docs/schemas.
        Default is None
        :return:
        """
        job_config = bigquery.job.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE"
        )
        if schema is None:
            job_config.autodetect = True
        else:
            job_config.skip_leading_rows = 1

        destination = self.get_table_ref(table_name=bq_table_name)

        job = self.bq_client.load_table_from_uri(
            source_uris=source_uris,
            destination=destination,
            job_config=job_config)

        job.result()

    # def set_expiration_time(self, table_name: str):
    #     """
    #     INCOMPLETE
    #     # https://cloud.google.com/bigquery/docs/samples/bigquery-update-table-expiration
    #     :param table_name: name of the table to be set an expiration time
    #     :return:
    #     """
    #     dataset_ref = bigquery.DatasetReference(self.project_name, self.dataset_name)
    #     table_ref = dataset_ref.table(table_id=table_name)
    #     # TODO: complete the method
    #     pass

    def query_to_dataframe(self, query):
        """
        :param query: str: SQL query to be executed
        :return: pandas.DataFrame
        """
        if self.check_query_validity(query=query):
            return self.bq_client.query(query=query).to_dataframe()

        raise BadRequest("""No dataframe is returned since the following query
         is incorrect :
         {query}
         """.format(query=query))

    def bq_table_to_dataframe(self, table_name: str):
        """
        :param table_name: name of the table to download to a dataframe
        :return: pandas.Dataframe
        """
        dataset_ref = bigquery.DatasetReference(self.project_name, self.dataset_name)
        table_ref = dataset_ref.table(table_id=table_name)
        try:
            table = self.bq_client.get_table(table_ref)
            return self.bq_client.list_rows(table).to_dataframe()
        except NotFound:
            return None

    def dataframe_to_bq(self, dataframe: pd.DataFrame, table_name: str, schema: list = None):
        """
        :param dataframe: pandas.DataFrame to be loaded in BQ
        :param table_name: name of the table
        :param schema: list of bigquery.SchemaField. Default is None
        :return: None
        """
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE"
        )

        table_id = '.'.join([self.project_name, self.dataset_name, table_name])

        job = self.bq_client.load_table_from_dataframe(
            dataframe=dataframe, destination=table_id,
            job_config=job_config
        )
        job.result()

    def copy_bq_table(self, source_table_id: str, destination_table_id: str):
        """
        :param source_table_id: bigquery ID ("your-project.source_dataset.source_table") of the table to be copied
        :param destination_table_id: bigquery ID ("your-project.destination_dataset.destination_table") of the table
        where to write the copy
        :return:
        """
        # (developer): Set source_table_id to the ID of the original table.
        # source_table_id = "your-project.source_dataset.source_table"

        # (developer): Set destination_table_id to the ID of the
        # destination_table_id =
        # "your-project.destination_dataset.destination_table"

        # delete existing table if exists
        self.delete_bq_table(table_id=destination_table_id)
        job = self.bq_client.copy_table(source_table_id, destination_table_id)
        job.result()

    def delete_bq_table(self, table_id: str):
        """
        :param table_id: bigquery ID ("your-project.source_dataset.your_table") of the table to be deleted
        :return:
        """
        self.bq_client.delete_table(table_id, not_found_ok=True)

    def rename_bq_table(self, source_table_id: str, new_table_name: str):
        """
        :param source_table_id: bigquery ID ("your-project.source_dataset.source_table") of the table to be renamed
        :param new_table_name: the new table name
        :return:
        """
        parts = source_table_id.split('.')
        parts[-1] = new_table_name
        destination_table_id = '.'.join(parts)
        self.copy_bq_table(source_table_id=source_table_id,
                           destination_table_id=destination_table_id
                           )
        self.delete_bq_table(table_id=source_table_id)

    def get_bq_client(self):
        """
        :return: bigquery.Client
        """
        if self.credentials is not None:
            return bigquery.Client(project=self.project_name,
                                   credentials=self.credentials)

        return bigquery.Client(project=self.project_name)

    def exists(self, table_name: str):
        """
        Check the existence of table_name at the default project_name.dataset_name
        :param table_name: name of the table
        :return: True if table_name exists in the default project_name.dataset_name else False
        """
        table_id = '.'.join([self.project_name, self.dataset_name, table_name])
        try:
            self.bq_client.get_table(table_id)  # Make an API request.
            return True
        except NotFound:
            return False

    def insert_row(self, table_name: str, row: list):
        """
        :param table_name:  name of the table where the row has to be
        inserted
        :param row: list of tuples of the form (column_name, data)
        :return:
        """
        table_id = '.'.join([self.project_name, self.dataset_name, table_name])

        # prepare rows
        rows_to_insert = [
            {"{col_name}".format(col_name=col_name): "{data}".format(
                data=data) if isinstance(data, str) else data
             for col_name, data in row
             }
        ]
        try:
            errors = self.bq_client.insert_rows_json(
                table_id,
                rows_to_insert)  # Make an API request.

            if len(errors) > 0:
                logger.error(
                    f"Encountered errors while inserting rows: {errors}"
                )

        except NotFound:
            logger.error(f"Could NOT find {table_id}")

    def query_cost(self, query: str):
        """
        :param query:
        :return: the cost generated by the execution of the query
        """
        job_config = bigquery.QueryJobConfig(dry_run=True,
                                             use_query_cache=False)

        try:
            query_job = self.bq_client.query(query=query,
                                             job_config=job_config)

            # cost is 5 dollars per TB
            # from BigQuery documentation :
            # """"
            # Charges are rounded up to the nearest MB, with a minimum
            # 10 MB data processed per table referenced by the query,
            # and with a minimum 20 MB data processed per query
            # """"
            processed_bytes = max(query_job.total_bytes_processed,
                                  20*(1024**2)
                                  )
            return 5*processed_bytes/(1024**4)

        except BadRequest:
            logger.warning(
                """This query is incorrect. Cannot estimate the cost.
                Returns 0 as default cost""")
            return 0

    def check_query_validity(self, query: str):
        """
        Performs a Dry Run to check if the query is correct
        :param query:
        :return: True if no errors in the query else False
        """
        job_config = bigquery.QueryJobConfig(dry_run=True,
                                             use_query_cache=False)

        # Start the query, passing in the extra configuration.
        try:
            _ = self.bq_client.query(query=query, job_config=job_config)

            # A dry run query completes immediately.
            # logger.info("This query will process {} bytes.".format(
            #     query_job.total_bytes_processed))
            return True

        except BadRequest:
            logger.warning("This query is incorrect. Will not run for real")
            return False

    def create_time_partition_table(self,
                                    table_name: str,
                                    schema_def: dict,
                                    expiration_nb_days: int = None,
                                    partition_column_name: str = None):
        """
        :param table_name:
        :param schema_def:
        see https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema.FIELDS.mode
        :param expiration_nb_days: number of days before partition expires
        :param partition_column_name: name of the column to perform the time partition
        """
        table_id = "{project_name}.{dataset_name}.{table_name}".format(project_name=self.project_name,
                                                                       dataset_name=self.dataset_name,
                                                                       table_name=table_name)
        schema = [bigquery.SchemaField("{column_name}".format(column_name=column_name),
                                       "{column_type}".format(column_type=column_type))
                  for column_name, column_type in schema_def.items()]

        table = bigquery.Table(table_id, schema=schema)

        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_column_name,  # name of column to use for partitioning
            expiration_ms=expiration_nb_days*24*60*60 if expiration_nb_days is not None else expiration_nb_days,
        )  # 90 days

        table = self.bq_client.create_table(table)  # Make an API request.
        logger.info(
            "Table {}.{}.{} is now created".format(table.project, table.dataset_id, table.table_id)
        )

    @staticmethod
    def get_bq_extract_job_config(avro_format: bool = False):
        """
        :param avro_format: If True load bq_table in avro format else
        csv. Default is False
        :return: bigquery.job.ExtractJobConfig
        """
        job_config = bigquery.job.ExtractJobConfig()
        if avro_format:
            job_config.destination_format = bigquery.DestinationFormat.AVRO
            # bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
            # bigquery.DestinationFormat.AVRO
        return job_config
