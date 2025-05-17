"""Functions to interact with AWS services"""

import json
import re
import io
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_appconfig_configuration(application, environment, configuration_profile):
    """
    Retrieve the application configuration from AWS AppConfig.

    This function starts a configuration session and retrieves the latest configuration 
    data for the specified application, environment, and configuration profile.

    Args:
        application (str): The identifier for the application.
        environment (str): The identifier for the environment.
        configuration_profile (str): The identifier for the configuration profile.

    Returns:
        dict: The configuration data as a dictionary.
    """
    client = boto3.client("appconfigdata")

    # Start the configuration session
    response = client.start_configuration_session(
        ApplicationIdentifier=application,
        EnvironmentIdentifier=environment,
        ConfigurationProfileIdentifier=configuration_profile,
    )
    session_token = response["InitialConfigurationToken"]

    # Retrieve the configuration data
    config_response = client.get_latest_configuration(ConfigurationToken=session_token)
    config_data = config_response["Configuration"].read().decode("utf-8")
    return json.loads(config_data)


def get_all_files(bucket_name, base_path="raw/", file_ending=".parquet"):
    """
    Retrieve all lap files stored in the raw layer of the specified S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        base_path (str): The base path in the bucket where lap files are stored (default is 'raw/').

    Returns:
        list: A list of S3 object keys (paths) for all lap files in the raw layer.
    """
    s3 = boto3.client("s3")
    lap_files = []

    # List all objects in the specified base path
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=base_path)

    # Check if the response contains any objects
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            # Filter for lap files (parquet files in the raw layer)
            if key.endswith(file_ending):
                lap_files.append(key)

    return lap_files


def parse_s3_path_to_dataframe(s3_path):
    """
    Parse the S3 path of lap files into a DataFrame with
      columns: season, eventname, session, and filename.

    Args:
        s3_path (str): The S3 path of the lap file.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed information.

    Raises:
        ValueError: If the provided S3 path does not match the expected format.
    """
    # Extract components from the S3 path
    match = re.match(r"raw/(\d{4})/([^/]+)/([^/]+)/(.+)", s3_path)
    if match:
        season = match.group(1)
        eventname = match.group(2).replace("_", " ").title()  # Convert to title case
        session = match.group(3).replace("_", " ").title()  # Convert to title case
        filename = match.group(4)

        # Create a DataFrame from the extracted components
        data = {
            "seasonyear": [season],
            "eventname": [eventname],
            "session": [session],
            "filename": [filename],
        }
        return pd.DataFrame(data)

    # Removed else: just return after raising
    raise ValueError("The provided S3 path does not match the expected format.")


def save_dataframe_as_parquet_to_s3(dataframe, bucket_name, target_path, file_name):
    """
    Save a DataFrame as a Parquet file to an S3 bucket.

    This function checks for duplicate column names in the DataFrame, converts the DataFrame 
    to a Parquet format, and uploads it to the specified S3 bucket.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        bucket_name (str): The name of the S3 bucket.
        target_path (str): The target path in the bucket where the file will be saved.
        file_name (str): The name of the file to save.

    Raises:
        ValueError: If the DataFrame contains duplicate column names.
    """
    # Check for duplicate column names
    if dataframe.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names.")

    # Convert dataframe to Parquet
    table = pa.Table.from_pandas(dataframe)
    parquet_buffer = io.BytesIO()
    pq.write_table(table, parquet_buffer)

    # Set the object key for S3
    object_key = target_path + file_name

    # Upload Parquet to S3
    s3 = boto3.client("s3")
    s3.put_object(Body=parquet_buffer.getvalue(), Bucket=bucket_name, Key=object_key)
    print(f"Uploaded parquet file to S3 {target_path}")


def get_all_filepaths_to_df(file_paths):
    """
    Parse a list of S3 file paths into a single DataFrame.

    This function processes each file path, converting it into a DataFrame and 
    concatenating all DataFrames into one.

    Args:
        file_paths (list): A list of S3 file paths to parse.

    Returns:
        pd.DataFrame: A DataFrame containing parsed information from all file paths.
    """
    parsed_file_paths = []
    for file in file_paths:
        file_path_df = parse_s3_path_to_dataframe(file)
        file_path_df["seasonyear"] = file_path_df["seasonyear"].astype(int)
        parsed_file_paths.append(file_path_df)

    return pd.concat(parsed_file_paths, ignore_index=True)
