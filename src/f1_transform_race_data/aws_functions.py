"""Functions to interact with AWS services."""

import io
import json
import re

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_appconfig_configuration(application, environment, configuration_profile):
    """
    Retrieve configuration data from AWS AppConfig.

    Starts a configuration session and fetches the latest configuration data
    for the specified application, environment, and configuration profile.

    Args:
        application (str): Application identifier.
        environment (str): Environment identifier.
        configuration_profile (str): Configuration profile identifier.

    Returns:
        dict: Parsed JSON configuration data.
    """
    client = boto3.client('appconfigdata')

    # Start the configuration session
    response = client.start_configuration_session(
        ApplicationIdentifier=application,
        EnvironmentIdentifier=environment,
        ConfigurationProfileIdentifier=configuration_profile,
    )
    session_token = response['InitialConfigurationToken']

    # Retrieve the configuration data
    config_response = client.get_latest_configuration(
        ConfigurationToken=session_token
    )
    config_data = config_response['Configuration'].read().decode('utf-8')
    return json.loads(config_data)


def get_all_files(bucket_name, base_path='raw/', file_ending='.parquet'):
    """
    Retrieve all lap files stored in the raw layer of the specified S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        base_path (str): The base path in the bucket where lap files are stored.
        file_ending (str): File extension to filter by (default: '.parquet').

    Returns:
        list: List of S3 object keys (paths) for all lap files in the raw layer.
    """
    s3 = boto3.client('s3')
    files = []

    # List all objects in the specified base path
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=base_path)

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            # Filter for lap files (parquet files in the raw layer)
            if key.endswith(file_ending):
                files.append(key)

    return files


def parse_s3_path_to_dataframe(s3_path):
    """
    Parse the S3 path of lap files into a DataFrame with season, eventname,
    session, and filename columns.

    Args:
        s3_path (str): The S3 path of the lap file.

    Returns:
        pd.DataFrame: DataFrame containing parsed information.

    Raises:
        ValueError: If the S3 path does not match the expected format.
    """
    match = re.match(r'raw/(\d{4})/([^/]+)/([^/]+)/(.+)', s3_path)
    if not match:
        raise ValueError("The provided S3 path does not match the expected format.")

    season = match.group(1)
    eventname = match.group(2).replace('_', ' ').title()
    session = match.group(3).replace('_', ' ').title()
    filename = match.group(4)

    data = {
        'seasonyear': [season],
        'eventname': [eventname],
        'session': [session],
        'filename': [filename],
    }
    return pd.DataFrame(data)


def save_dataframe_as_parquet_to_s3(dataframe, bucket_name, target_path, file_name):
    """
    Save a pandas DataFrame as a Parquet file to a specified S3 location.

    Args:
        dataframe (pd.DataFrame): DataFrame to save.
        bucket_name (str): Target S3 bucket name.
        target_path (str): Path inside the bucket to save the file.
        file_name (str): Name of the Parquet file.

    Raises:
        ValueError: If the DataFrame contains duplicate column names.
    """
    if dataframe.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names.")

    table = pa.Table.from_pandas(dataframe)
    parquet_buffer = io.BytesIO()
    pq.write_table(table, parquet_buffer)

    object_key = target_path + file_name

    s3 = boto3.client('s3')
    s3.put_object(
        Body=parquet_buffer.getvalue(),
        Bucket=bucket_name,
        Key=object_key,
    )
    print(f'Uploaded parquet file to S3 {target_path}')


def get_all_filepaths_to_df(file_paths):
    """
    Convert a list of S3 file paths into a concatenated DataFrame with parsed components.

    Args:
        file_paths (list): List of S3 file path strings.

    Returns:
        pd.DataFrame: Concatenated DataFrame with parsed file path components.
    """
    parsed_file_paths = []
    for file in file_paths:
        file_path_df = parse_s3_path_to_dataframe(file)
        file_path_df['seasonyear'] = file_path_df['seasonyear'].astype(int)
        parsed_file_paths.append(file_path_df)

    return pd.concat(parsed_file_paths, ignore_index=True)


def read_parquet_files_from_s3(bucket_name, file_paths):
    """
    Read multiple Parquet files from specified S3 file paths.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_paths (list): List of file paths to read Parquet files from.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all parquet files' data.

    Notes:
        Files that are missing or cause errors will be skipped with a warning.
    """
    s3 = boto3.client('s3')
    parquet_data_list = []

    for file_path in file_paths:
        try:
            response = s3.get_object(Bucket=bucket_name, Key=file_path)
            content = response['Body'].read()

            parquet_data = pd.read_parquet(io.BytesIO(content))
            parquet_data_list.append(parquet_data)

        except s3.exceptions.NoSuchKey:
            print(f"File not found: {file_path}")

        except Exception as e:  # Broad exception catch, consider specific exceptions if possible
            print(f"Error reading Parquet file: {file_path}")
            print(f"Error: {str(e)}")

    try:
        return pd.concat(parquet_data_list)
    except Exception as e:
        print(f"Error concatenating DataFrames: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure
