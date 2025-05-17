"""AWS Lambda fucntion to transform data """

import json
import os
import requests
import pandas as pd
from aws_functions import get_all_files, save_dataframe_as_parquet_to_s3, read_parquet_files_from_s3
from f1_transformations import transform_lap_sector_time_columns, transform_qualifying_data, transform_race_data, get_speed_aggregates, pivot_practice_data_to_features, get_race_speed_aggregates
from f1_transformations import get_driver_standings, get_team_standings, get_earliest_latest_session_dateutc, combine_data


def send_reply(chat_id, message, bot_token):
    """
    Sends a reply message to the specified Telegram chat.

    Args:
        chat_id (int): The Telegram chat ID.
        message (str): The message to send.
        bot_token (str): The Telegram bot token.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }

    try:
        # Set a timeout of 5 seconds for the request
        response = requests.post(url, json=payload, timeout=5)

        if response.status_code != 200:
            print(f"Failed to send message to chat_id {chat_id}: {response.text}")
        else:
            print(f"Sent message to chat_id {chat_id}: {message}")
    except requests.exceptions.Timeout:
        print(f"Request to chat_id {chat_id} timed out.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")



def add_event_date(race_data, race_schedule):
    """Merge race data with event dates from the race schedule.

    This function takes two DataFrames: race_data and race_schedule. 
    It merges them on the 'seasonyear' and 'eventname' columns to 
    include event dates and session dates in the race data.

    Args:
        race_data (DataFrame): A DataFrame containing race information.
        race_schedule (DataFrame): A DataFrame containing event names, 
                                   event dates, and session dates.

    Returns:
        DataFrame: A new DataFrame that includes the original race data 
                    along with the corresponding event dates and session 
                    dates from the race schedule.
    """
    race_data_with_date = race_data.merge(race_schedule[['seasonyear','eventname',
                                                          'eventdate','sessiondateutc']],
                                          on=['seasonyear','eventname'],
                                          how='left')
    return race_data_with_date

def lambda_handler(event, context):
    """AWS Lambda handler fortransforming F1 race data, creating features,
    and storing transformed parquet files to S3.

    Args:
        event (dict): Lambda event payload (not used).
        context (object): Lambda context object (not used).

    Returns:
        dict: Response containing status code and success message.
    """
    bot_token = os.environ['bot_token']
    chat_id = os.environ['chat_id']
    bucket_name = event['bucket_name']
    base_path = event['layer']
    # Schedule
    schedule_files = get_all_files(bucket_name=bucket_name,
                                   base_path=base_path,
                                   file_ending='schedules.parquet')
    schedule = read_parquet_files_from_s3(bucket_name,
                                          file_paths=schedule_files)
    race_schedule = schedule[schedule['session']=='Race']
    # Laps
    lap_files = get_all_files(bucket_name=bucket_name,
                              base_path=base_path,
                              file_ending='_laps.parquet')
    laps_raw = read_parquet_files_from_s3(bucket_name,
                                          file_paths=lap_files)
    # Qualifying Results
    qualifying_result_files = get_all_files(bucket_name=bucket_name,
                                            base_path=base_path,
                                            file_ending='qualifying_results.parquet')
    qualifying_results_raw = read_parquet_files_from_s3(bucket_name,
                                                        file_paths=qualifying_result_files)
    # Race Results
    race_result_files = get_all_files(bucket_name=bucket_name,
                                      base_path=base_path,
                                      file_ending='race_results.parquet')
    race_results_raw = read_parquet_files_from_s3(bucket_name,
                                                  file_paths=race_result_files)

    # Transformation
    lap_time_columns = ['laptime','pitouttime','pitintime',
                        'sector1time','sector2time','sector3time']
    laps = laps_raw.pipe(transform_lap_sector_time_columns, lap_time_columns)
    speed_aggregates = get_speed_aggregates(laps)

    practice_speed_aggregates = speed_aggregates[speed_aggregates['session']\
                                                    .isin(['Practice 1','Practice 2','Practice 3'])]
    practice_data = pivot_practice_data_to_features(practice_speed_aggregates)
    race_speed_aggregates = get_race_speed_aggregates(speed_aggregates)

    qualifying_data = transform_qualifying_data(qualifying_results_raw)
    race_results = race_results_raw.pipe(transform_race_data)

    race_weekend_dates = get_earliest_latest_session_dateutc(schedule)
    driver_standings = get_driver_standings(race_results, race_weekend_dates)
    team_standings = get_team_standings(race_results, race_weekend_dates)
    driver_team_standings = driver_standings.merge(team_standings,
                                                   on=['seasonyear', 'roundnumber',
                                                       'eventname', 'teamid'],
                                                   how='left')

    nullable_columns = ['q1_s','q2_s', 'q3_s', 'p3_num_laps', 'p3_speedi1_max',
                        'p3_speedi2_max', 'p3_speedfl_max', 'p3_speedi1_mean',
                        'p3_speedi2_mean', 'p3_speedfl_mean', 'p3_speedst_mean']
    pract_qual_with_standings_race_results_agg = combine_data(practice_data,
                                                              qualifying_data,
                                                              driver_team_standings,
                                                              race_results,
                                                              race_speed_aggregates,
                                                              nullable_columns)

    # Create dummies for greand prix and team
    gp_dummies = pd.get_dummies(pract_qual_with_standings_race_results_agg['eventname'],
                                prefix='event')
    gp_dummies.columns = gp_dummies.columns.str.lower()

    team_dummies = pd.get_dummies(pract_qual_with_standings_race_results_agg['teamid'],
                                  prefix='team')
    team_dummies.columns = team_dummies.columns.str.lower()

    race_data_w_gp_dummies = pd.concat([pract_qual_with_standings_race_results_agg,
                                        gp_dummies], axis=1)
    race_data = pd.concat([race_data_w_gp_dummies, team_dummies], axis=1)

    race_data = race_data.pipe(add_event_date, race_schedule)
    save_dataframe_as_parquet_to_s3(race_data,
                                    bucket_name='f1-race-prediction',
                                    target_path='transformed/',
                                    file_name='transformed_race_data.parquet')
    send_reply(chat_id=chat_id,
               message=f"F1 Data Transformation successful, processed rows: {len(race_data)}",
               bot_token=bot_token)

    return {
        'statusCode': 200,
        'body': json.dumps('Success')
    }
