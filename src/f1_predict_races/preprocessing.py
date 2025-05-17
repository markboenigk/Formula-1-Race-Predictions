"""Preprocessing functions for race prediction data pipeline."""

from aws_functions import get_all_files, read_parquet_files_from_s3


def load_and_prepare_data(bucket_name, base_path):
    """Loads and prepares race data from S3.

    Args:
        bucket_name (str): Name of the S3 bucket.
        base_path (str): Base path in the S3 bucket where the files are located.

    Returns:
        pd.DataFrame: Combined DataFrame containing all race data.
    """
    files = get_all_files(bucket_name=bucket_name,
                          base_path=base_path,
                          file_ending='transformed_race_data.parquet')
    data = read_parquet_files_from_s3(bucket_name, file_paths=files)
    return data


def get_date_of_to_predict_race(race_data, race_to_predict_gp, race_to_predict_year):
    """Retrieves the event date for the race to be predicted.

    Args:
        race_data (pd.DataFrame): DataFrame containing race event information.
        race_to_predict_gp (str): Name of the Grand Prix to predict.
        race_to_predict_year (int): Season year of the race.

    Returns:
        pd.Timestamp: Date of the specified race.
    """
    print(f"Predicting race: {race_to_predict_gp} {race_to_predict_year}")
    race_date = race_data[
        (race_data['eventname'] == race_to_predict_gp) &
        (race_data['seasonyear'] == race_to_predict_year)
    ]['eventdate'].iloc[0]
    return race_date


def determine_target_race(data, race_name=None, race_season=None):
    """Determines the target race for prediction.

    If race name and season are not provided, the most recent race is used.

    Args:
        data (pd.DataFrame): Race data.
        race_name (str, optional): Name of the race to predict.
        race_season (int, optional): Season year of the race to predict.

    Returns:
        Tuple[str, int, pd.Timestamp]: Race name, season year, and event date.
    """
    try:
        if race_name and race_season:
            print('Got Race Name and season')
            date = get_date_of_to_predict_race(data, race_name, race_season)
        else:
            latest = data['eventdate'].max()
            race_name = data[data['eventdate'] == latest]['eventname'].iloc[0]
            race_season = data[data['eventdate'] == latest]['seasonyear'].iloc[0]
            date = latest
    except NameError as e:
        latest = data['eventdate'].max()
        race_name = data[data['eventdate'] == latest]['eventname'].iloc[0]
        race_season = data[data['eventdate'] == latest]['seasonyear'].iloc[0]
        date = latest
        print(f"Error determining target race: {e}. Using {race_name} {race_season} instead.")
    return race_name, race_season, date


def drop_race_y_columns(race_data, target_column, all_y_columns):
    """Drops all target columns except the specified one, along with 'race_num_laps'.

    Args:
        race_data (pd.DataFrame): Input race data.
        target_column (str): Target column to retain.
        all_y_columns (List[str]): List of all target columns.

    Returns:
        pd.DataFrame: DataFrame with only the specified target column retained.
    """
    race_data = race_data.copy()
    columns_to_drop = [col for col in all_y_columns if col != target_column] + ['race_num_laps']
    race_data = race_data.drop(columns=columns_to_drop)
    return race_data


def drop_columns_with_na(df, threshold):
    """Drops columns with a percentage of NA values above the given threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Percentage threshold (0 to 100).

    Returns:
        pd.DataFrame: DataFrame with high-NA columns removed.
    """
    na_threshold = len(df) * (threshold / 100)
    df_dropped = df.dropna(axis=1, thresh=len(df) - na_threshold)
    return df_dropped


def split_data_for_prediction(data, race_name, race_year, target_column, all_y_columns):
    """Splits the race data into training and prediction sets.

    Args:
        data (pd.DataFrame): Input race data.
        race_name (str): Name of the race to predict.
        race_year (int): Year of the race to predict.
        target_column (str): Target column to predict.
        all_y_columns (List[str]): List of all target columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training data and prediction data.
    """
    event_date = get_date_of_to_predict_race(data, race_name, race_year)
    filtered = data[data['eventdate'] <= event_date].sort_values(by='eventdate')
    cleaned = drop_race_y_columns(filtered, target_column, all_y_columns)\
                .pipe(drop_columns_with_na, threshold=90)

    train = cleaned[~((cleaned['seasonyear'] == race_year)
                       & (cleaned['eventname'] == race_name))].copy()
    predict = cleaned[(cleaned['seasonyear'] == race_year)
                       & (cleaned['eventname'] == race_name)].copy()
    train = train.dropna(subset=[target_column])
    return train, predict


def get_feature_sets(df, target_column):
    """Creates feature sets by excluding metadata and target columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Target column to exclude from features.

    Returns:
        Dict[str, List[str]]: Dictionary containing 'base_feature' and 'all_features'.
    """
    non_features = ['seasonyear', 'eventname', 'eventdate', 'sessiondateutc', 'driver',
                    'roundnumber', 'teamid', 'session', 'eventname_abbr', 'qualifying_data_exist', 
                    target_column]
    all_features = [col for col in df.columns if col not in non_features]
    return {'base_feature': ['qtime_s'], 'all_features': all_features}


def fill_na_with_zero(df):
    """Fills NaN values with 0 for columns starting with 'p' or 'q'.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with NaNs in 'p*' and 'q*' columns filled with 0.
    """
    cols_to_fill = df.columns[df.columns.str.startswith('p')]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    cols_to_fill = df.columns[df.columns.str.startswith('q')]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    return df
