# General packages
import argparse
import logging
import os
import sys
import yaml
import json
import warnings
import time
from datetime import datetime
from functools import reduce

# Google related packages
import google.auth as ga
from google.oauth2 import service_account

# Project packages
from gcp_interface.storage_interface import StorageInterface

from models.classifier import train_model_in_local, predict_in_local

warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")

NUMBER_ITERATION = 10
PARSER_TASK_CHOICE = ['train', 'predict']

DATA_CONFIG = {'features': {'fixed': ['Pclass', 'Age', 'SibSp', 'Parch'],
                            'age_col': 'Age',
                            'gender_col': 'Sex'
                            },
               'target_col': 'Survived',
               'passenger_id': 'PassengerId'
               }

if __name__ == '__main__':
    start = datetime.now()  # start of the script
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=LOGLEVEL)
    logger = logging.getLogger()
    # parse all given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str,
                        choices=PARSER_TASK_CHOICE)
    parser.add_argument('--configuration', required=True, type=str)
    parser.add_argument('--env', required=False, type=str, choices=['local', 'cloud'], default='local')

    args = parser.parse_args()

    # log input arguments:
    logger.info("Input arguments are :")
    for t, inpt in args.__dict__.items():
        logger.info(f"{t}: {inpt}")

    # retrieve infrastructure data and functional parameters
    with open(args.configuration, 'r') as f:
        config = yaml.safe_load(f)

    # retrieve credentials
    if config['google_cloud']['credentials_json_file'] != "":
        credentials = service_account.Credentials.from_service_account_file(
            config['google_cloud']['credentials_json_file'])
    else:
        credentials, _ = ga.default()

    # instantiate a GS interface
    gs_interface = StorageInterface(
        project_name=config['project_name'],
        credentials=credentials)

    # build the model name according to your need (e.g. by taking account of the current date or other information)
    model_name = 'titanic_rf_classifier'
    data_name = 'titanic_train' if args.task == 'train' else 'titanic_test'
    if args.task == 'train':
        # gs_dir_path = 'ai_platform_template_dir' = config.get('google_gcs').get('directory_name')
        train_model_in_local(
            gs_interface=gs_interface,
            gs_dir_path=config['google_gcs'].get('directory_name'),
            bucket_name=config['google_gcs'].get('bucket_name'),
            local_dir_path=config.get('local_dir_path', "tmp"),
            train_name=data_name,
            data_configuration=DATA_CONFIG,
            model_name=model_name
        )
    elif args.task == 'predict':
        predict_in_local(gs_interface=gs_interface,
                         gs_dir_path=config['google_gcs'].get('directory_name'),
                         bucket_name=config['google_gcs'].get('bucket_name'),
                         local_dir_path=config.get('local_dir_path', "tmp"),
                         predict_name=data_name,
                         data_configuration=DATA_CONFIG,
                         model_name=model_name
                         )

    else:
        pass

    logger.info(f"Task {args.task} is successfully completed.")
