"""Functions for interacting with AWS services."""
import io
import json
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_all_files(bucket_name, base_path='raw/', file_ending='.parquet'):
    """Retrieves all files with a specified file ending from a given path in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        base_path (str, optional): S3 prefix to search in (default is 'raw/').
        file_ending (str, optional): File extension to filter for (default is '.parquet').

    Returns:
        List[str]: List of S3 object keys matching the file ending.
    """
    s3 = boto3.client('s3')
    files = []

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=base_path)

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith(file_ending):
                files.append(key)

    return files


def read_parquet_files_from_s3(bucket_name, file_paths):
    """Reads multiple Parquet files from S3 and returns them as a single DataFrame.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_paths (List[str]): List of file paths to read from S3.

    Returns:
        pd.DataFrame: Concatenated DataFrame from all successfully read Parquet files.
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
        except (pd.errors.ParserError, ValueError, OSError) as e:
            print(f"Failed to read or parse Parquet file {file_path}: {str(e)}")

    if not parquet_data_list:
        raise ValueError("No valid Parquet files were read from S3.")

    return pd.concat(parquet_data_list)


def save_dataframe_as_parquet_to_s3(dataframe, bucket_name, target_path, file_name):
    """Saves a Pandas DataFrame as a Parquet file to an S3 bucket.

    Args:
        dataframe (pd.DataFrame): DataFrame to save.
        bucket_name (str): Name of the S3 bucket.
        target_path (str): Directory path in the S3 bucket.
        file_name (str): Name of the Parquet file (e.g., 'data.parquet').

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
    s3.put_object(Body=parquet_buffer.getvalue(),
                  Bucket=bucket_name,
                  Key=object_key)
    print(f'Uploaded parquet file to S3 {target_path}')


def save_json_to_s3(data, bucket_name, target_path, file_name):
    """Saves a dictionary or JSON-serializable object to S3 as a .json file.

    Args:
        data (dict): JSON-serializable object (e.g., a dictionary).
        bucket_name (str): Name of the S3 bucket.
        target_path (str): Directory path in the S3 bucket.
        file_name (str): Name of the JSON file (e.g., 'summary.json').
    """
    json_str = json.dumps(data, indent=4)
    json_buffer = io.BytesIO(json_str.encode('utf-8'))
    object_key = target_path + file_name

    s3 = boto3.client('s3')
    s3.put_object(Body=json_buffer.getvalue(),
                  Bucket=bucket_name,
                  Key=object_key,
                  ContentType='application/json')
    print(f'Uploaded JSON file to S3: {object_key}')
