"""AWS Lambda function to fetch Formula 1 data from FastF1 API and ingest it into S3"""
import json
import re
from datetime import datetime, timezone
import time

import pandas as pd
import numpy as np
import fastf1

from aws_functions import get_all_files, save_dataframe_as_parquet_to_s3
from aws_functions import get_appconfig_configuration, get_all_filepaths_to_df
from f1_functions import get_schedules, transform_schedules


def transform_column_names(df):
    """Convert DataFrame column names to lowercase with underscores and no special characters.

    Args:
        df (pd.DataFrame): Input DataFrame whose columns will be transformed.

    Returns:
        pd.DataFrame: DataFrame with transformed column names.
    """
    def convert_name(name):
        """Convert a single column name to snake_case and lowercase.

        Args:
            name (str): Original column name.

        Returns:
            str: Transformed column name.
        """
        name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)  # Remove special characters
        return name.lower()

    df.columns = [convert_name(col) for col in df.columns]
    return df


def clean_name_for_storage(name):
    """Clean a string by replacing spaces with underscores and removing special characters.

    Args:
        name (str): Original string.

    Returns:
        str: Cleaned, lowercased string suitable for file paths or keys.
    """
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w_]', '', name)
    return name.lower()

def save_session_dataframe(df, row, bucket_name, base_path, suffix):
    """
    Add metadata, transform columns, build file name, and save dataframe as parquet to S3.

    Args:
        df (pd.DataFrame): DataFrame to save.
        row (pd.Series): Row with session metadata.
        bucket_name (str): Name of S3 bucket.
        base_path (str): Base path inside S3 bucket.
        suffix (str): File suffix like 'laps', 'results', or 'session_info'.
    """
    df['seasonyear'] = row['seasonyear']
    df['eventname'] = row['eventname']
    df['session'] = row['session']
    df = transform_column_names(df)
    season = row['seasonyear']
    event = clean_name_for_storage(row['eventname'])
    session_name = clean_name_for_storage(row['session'])
    file_name = (
        f"{season}/{event}/{session_name}/"
        f"{season}_{event}_{session_name}_{suffix}.parquet"
    )
    save_dataframe_as_parquet_to_s3(df, bucket_name, base_path, file_name)


def process_and_store_sessions(race_data, bucket_name, base_path):
    """
    Process each race session from the provided DataFrame,
      transform data, and save as parquet files in S3.

    For each session in the race_data DataFrame, this function:
    - Loads session data using fastf1
    - Extracts lap data, qualifying results, race results, and session info where applicable
    - Adds metadata columns and transforms column names
    - Saves processed data as parquet files to the specified S3 bucket path
    - Sleeps 21 seconds between sessions to avoid rate limiting

    Args:
        race_data (pd.DataFrame): DataFrame with columns
          including 'seasonyear', 'eventname', and 'session'.
        bucket_name (str): Name of the target S3 bucket.
        base_path (str): Base directory path inside the S3 bucket where files will be stored.

    Raises:
        Exception: Any exceptions during processing are caught and printed but do not stop the loop.
    """
    for _, row in race_data.iterrows():
        try:
            print(f"Processing {row['seasonyear']} {row['eventname']} {row['session']}")
            session = fastf1.get_session(row['seasonyear'], row['eventname'], row['session'])
            session.load()

            lap_data = session.laps
            if lap_data is not None and not lap_data.empty:
                save_session_dataframe(lap_data, row, bucket_name, base_path, 'laps')

            if row['session'] == 'Qualifying':
                session_results = session.results
                if session_results is not None and not session_results.empty:
                    save_session_dataframe(session_results,
                                            row,
                                            bucket_name,
                                            base_path,
                                            'results')

            if row['session'] == 'Race':
                session_results = session.results
                if session_results is not None and not session_results.empty:
                    save_session_dataframe(session_results,
                                           row,
                                           bucket_name,
                                           base_path,
                                           'results')

                session_info = pd.json_normalize(session.session_info)
                if session_info is not None and not session_info.empty:
                    print('Session', session_info)
                    save_session_dataframe(session_info,
                                            row,
                                            bucket_name,
                                            base_path,
                                            'session_info')

            time.sleep(21)

        except Exception as e:
            print(f"Error processing session {row['eventname']} {row['session']}: {e}")

        fastf1.Cache.clear_cache()


def get_lap_file_paths(bucket_name='', base_path='', file_ending='_laps.parquet'):
    """Retrieve all lap file paths from S3 and return as a DataFrame.

    Args:
        bucket_name (str): S3 bucket name to query.
        base_path (str): Path prefix inside the bucket.
        file_ending (str): File name suffix to filter lap files.

    Returns:
        pd.DataFrame: DataFrame with columns including 'lap_file_name' and 'lap_file_exist'.
    """
    lap_files = get_all_files(bucket_name=bucket_name, base_path=base_path, file_ending=file_ending)
    lap_file_paths_df = get_all_filepaths_to_df(lap_files)
    lap_file_paths_df = lap_file_paths_df.rename(columns={'filename': 'lap_file_name'})
    return lap_file_paths_df


def get_session_results_file_paths(bucket_name='', base_path='', file_ending='_results.parquet'):
    """Retrieve all session results file paths from S3 and return as a DataFrame.

    Args:
        bucket_name (str): S3 bucket name to query.
        base_path (str): Path prefix inside the bucket.
        file_ending (str): File name suffix to filter results files.

    Returns:
        pd.DataFrame: DataFrame with columns including 
        'session_result_file_name' and 'session_results_exist'.
    """
    session_result_files = get_all_files(bucket_name=bucket_name,
                                          base_path=base_path, file_ending=file_ending)
    session_results_file_paths_df = get_all_filepaths_to_df(session_result_files)
    session_results_file_paths_df = session_results_file_paths_df.rename(
                                        columns={'filename': 'session_result_file_name'})
    return session_results_file_paths_df


def add_file_checks_to_schedule(schedule, laps_file_check, session_results_file_check):
    """Merge schedule DataFrame with lap and session results file checks to flag data existence.

    Args:
        schedule (pd.DataFrame): Schedule data containing 'seasonyear', 'eventname', 'session'.
        laps_file_check (pd.DataFrame): DataFrame indicating lap file existence.
        session_results_file_check (pd.DataFrame): DataFrame with session results.

    Returns:
        pd.DataFrame: Schedule DataFrame enhanced with columns for lap file existence,
                      session results existence, and a combined 'all_files_exist' flag.
    """
    schedule_w_laps_file_check = pd.merge(schedule,
                                           laps_file_check,
                                           on=['seasonyear', 'eventname', 'session'],
                                           how='left')
    schedule_w_laps_results_file_check = pd.merge(schedule_w_laps_file_check,
                                                   session_results_file_check,
                                                   on=['seasonyear', 'eventname', 'session'],
                                                   how='left')
    schedule_w_laps_results_file_check['lap_file_exist'] = schedule_w_laps_results_file_check['lap_file_exist'].fillna(False)
    schedule_w_laps_results_file_check['session_results_exist'] = schedule_w_laps_results_file_check['session_results_exist'].fillna(False)
    schedule_w_laps_results_file_check['all_files_exist'] = False
    schedule_w_laps_results_file_check['all_files_exist'] = np.where(
        (schedule_w_laps_results_file_check['lap_file_exist'] is True) &
        (schedule_w_laps_results_file_check['session_results_exist'] is True),
        True,
        False)
    schedule_w_laps_results_file_check['all_files_exist'] = np.where(
        (~schedule_w_laps_results_file_check['session'].isin(['Qualifying', 'Race'])) &
        (schedule_w_laps_results_file_check['lap_file_exist'] is True),
        True,
        schedule_w_laps_results_file_check['all_files_exist'])

    return schedule_w_laps_results_file_check


def lambda_handler(event, context):
    """AWS Lambda handler for ingesting F1 race data, processing session data,
    and storing transformed parquet files to S3.

    Args:
        event (dict): Lambda event payload (not used).
        context (object): Lambda context object (not used).

    Returns:
        dict: Response containing status code and success message.
    """
    application = "<your-appconfig-application-name>"
    environment = "<your-environment>"
    configuration_profile = "<your configuration-profile>"
    fastf1.Cache.enable_cache('/tmp')

    # Retrieve configuration from AppConfig
    config = get_appconfig_configuration(application, environment, configuration_profile)
    print("Retrieved Configuration:", config)
    seasons = config['seasons_to_ingest']
    schedules = get_schedules(seasons)
    schedules_t = schedules.pipe(transform_schedules).pipe(transform_column_names)
    schedules_t = schedules_t[schedules_t['eventname'] != 'Pre-Season Test']
    save_dataframe_as_parquet_to_s3(schedules_t,
                                     bucket_name='f1-race-prediction',
                                     target_path='raw/',
                                     file_name='schedules.parquet')

    # Check existing files in S3
    lap_files = get_lap_file_paths(bucket_name='<your-bucket-name>',
                                    base_path='<your-directory>',
                                    file_ending='_laps.parquet')
    session_results_files = get_session_results_file_paths(bucket_name='<your-bucket-name>',
                                                            base_path='<your-directory>',
                                                            file_ending='_results.parquet')

    schedules_trans_w_file_checks = add_file_checks_to_schedule(schedules_t,
                                                                       lap_files,
                                                                       session_results_files)
    sessions_wo_existing_data = schedules_trans_w_file_checks[schedules_trans_w_file_checks['all_files_exist'] is False]

    current_utc_time = pd.to_datetime(datetime.now(timezone.utc))
    past_sessions_wo_data = sessions_wo_existing_data[sessions_wo_existing_data['sessiondateutc'] < current_utc_time]

    # Filter out testing sessions and only process the latest season
    past_sessions_wo_data = past_sessions_wo_data[
        (past_sessions_wo_data['eventformat'] != 'testing') &
        (past_sessions_wo_data['seasonyear'] == past_sessions_wo_data['seasonyear'].max())
    ]

    process_and_store_sessions(past_sessions_wo_data,
                                bucket_name='<your-bucket-name>',
                                base_path='<your-directory>')

    return {
        'statusCode': 200,
        'body': json.dumps('Success!')
    }
