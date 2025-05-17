"""AWS Lambda function for F1 race prediction using machine learning models."""

import json
import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from preprocessing import (
    load_and_prepare_data,
    determine_target_race,
    split_data_for_prediction,
    get_feature_sets,
    fill_na_with_zero,
)
from model_training_prediction import train_and_evaluate_models, summarize_and_predict
from aws_functions import save_dataframe_as_parquet_to_s3, save_json_to_s3


def lambda_handler(event, context):
    """
    Lambda function entry point for training models, making predictions, and saving results to S3.

    Args:
        event (dict): Event data with keys 'gp_name', 'gp_season', 'target_column'.
        context (object): Lambda context (unused).

    Returns:
        dict: Response with status and prediction metadata.
    """
    bucket = "<your-aws-bucket>"
    base_path = "<your-directory>"
    race_name = event["gp_name"]
    race_season = event["gp_season"]
    target_column = event["target_column"]

    data = load_and_prepare_data(bucket, base_path)
    race_name, race_season, _ = determine_target_race(data, race_name, race_season)

    print(race_name)

    pot_y = {
        'regression': ['race_time_s', 'points', 'race_laptime_s_min',
                        'race_laptime_s_mean', 'calc_race_avg_laptime'],
        'classification': ['status', 'finished', 'finalposition']
    }
    y_columns = [item for sublist in pot_y.values() for item in sublist]

    train_df, predict_df = split_data_for_prediction(data, race_name,
                                                     race_season, target_column,
                                                     y_columns)
    train_df = fill_na_with_zero(train_df)
    features = get_feature_sets(train_df, target_column)

    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }

    results = train_and_evaluate_models(train_df, target_column, features, models)
    metadata, prediction_df, top_features = summarize_and_predict(results,
                                                                   predict_df,
                                                                   top_n=10,
                                                                   target_column=target_column)

    # Store data
    target_layer = 'predictions/'
    directory = f"{target_layer}{race_season}/{race_name}/"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dataframe_as_parquet_to_s3(
        prediction_df,
        bucket_name=bucket,
        target_path=directory,
        file_name=f"{timestamp}_{race_name}_{race_season}_predictions.parquet",
    )
    save_dataframe_as_parquet_to_s3(
        top_features,
        bucket_name=bucket,
        target_path=directory,
        file_name=f"{timestamp}_{race_name}_{race_season}_top_features.parquet",
    )
    metadata["race"] = f"{race_name} {race_season}"
    metadata["run_timestamp"] = timestamp
    save_json_to_s3(
        metadata,
        bucket_name=bucket,
        target_path=directory,
        file_name=f"{timestamp}_{race_name}_{race_season}_metadata.json",
    )
    return {
    'statusCode': 200,
    'body': json.dumps({
        'status': 'success',
        'predictions': prediction_df.to_dict(orient='records'), 
        'metadata': {
            'race': f"{race_name} {race_season}",
            'run_timestamp': timestamp,
            'top_features': top_features.to_dict(orient='records')
        }
    })
}
