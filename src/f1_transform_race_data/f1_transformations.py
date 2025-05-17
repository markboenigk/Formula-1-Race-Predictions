"""Functions for data transformation, create features and prepare for prediction"""
from itertools import filterfalse
from datetime import datetime, timezone 
import pandas as pd 
import numpy as np


def get_speed_aggregates(laps):
    """Compute mean and maximum speed-related metrics per driver and session.

    Args:
        df (pd.DataFrame): Input dataframe with speed, lap time, and position data.

    Returns:
        pd.DataFrame: Aggregated features per driver per session.
    """

    speed_aggregates_maximum = laps.groupby(['seasonyear','eventname','session','driver']).agg({'laptime_s':'min',
                                                                                                'lapnumber':'max',
                                                                                                'speedi1':'max',
                                                                                                'speedi2':'max',
                                                                                                'speedfl':'max',
                                                                                                'speedst':'max'}).reset_index()
    speed_aggregates_maximum = speed_aggregates_maximum.rename(columns={'laptime_s':'laptime_s_min',
                                                                        'lapnumber':'num_laps', 
                                                                        'speedi1':'speedi1_max', 
                                                                        'speedi2':'speedi2_max',
                                                                        'speedfl':'speedfl_max',
                                                                        'speedst':'speedst_max'})
    
    speed_aggregates_mean = laps.groupby(['seasonyear','eventname','session','driver']).agg({'laptime_s':'mean',
                                                                                            'speedi1':'mean',
                                                                                            'speedi2':'mean',
                                                                                            'speedfl':'mean',
                                                                                            'speedst':'mean'}).reset_index()
    speed_aggregates_mean = speed_aggregates_mean.rename(columns={'laptime_s':'laptime_s_mean','speedi1':'speedi1_mean','speedi2':'speedi2_mean',
                                                        'speedfl':'speedfl_mean','speedst':'speedst_mean'})

    speed_aggregates = speed_aggregates_maximum.merge(speed_aggregates_mean, on=['seasonyear', 'eventname', 'session', 'driver'])
    return speed_aggregates


def pivot_practice_data_to_features(practice_data):
    """Pivot practice session features into a single row per driver-event.

    Args:
        df (pd.DataFrame): Practice session features.

    Returns:
        pd.DataFrame: Pivoted practice features with FP1, FP2, FP3 columns.
    """
    # Column names transformation
    cols_to_drop_from_time_speed_cols = ['seasonyear','eventname',
                                         'session','driver']
    all_columns = list(practice_data.columns)
    time_speed_columns = list(filterfalse(lambda x: x in cols_to_drop_from_time_speed_cols,
                                           all_columns))

    # Practice 1
    p1_data = practice_data[practice_data['session'] == 'Practice 1']
    p1_cols = cols_to_drop_from_time_speed_cols + ['p1_' + col for col in time_speed_columns]

    p1_data.columns = p1_cols
    p1_data = p1_data.drop(columns=['session'])
    # Practice 2 
    p2_data = practice_data[practice_data['session'] == 'Practice 2']
    p2_cols = cols_to_drop_from_time_speed_cols + ['p2_' + col for col in time_speed_columns]
    p2_data.columns = p2_cols
    p2_data = p2_data.drop(columns=['session'])
    # Practice 3
    p3_data = practice_data[practice_data['session'] == 'Practice 3']
    p3_cols = cols_to_drop_from_time_speed_cols + ['p3_' + col for col in time_speed_columns]
    p3_data.columns = p3_cols
    p3_data = p3_data.drop(columns=['session'])
    # Combine P1, P2, P3 data 
    p1_2_data = p1_data.merge(p2_data, 
                              on=['seasonyear','eventname','driver'], 
                              how='left')
    p1_2_3_data = p1_2_data.merge(p3_data,
                                   on=['seasonyear', 'eventname', 'driver'], 
                                   how='left')
    return p1_2_3_data


def get_race_speed_aggregates(speed_aggregates):
    """Aggregate race speed-related metrics for each driver.

    Args:
        df (pd.DataFrame): Input dataframe with race data.

    Returns:
        pd.DataFrame: Aggregated race data with prefixed feature names.
    """
    race_speed_aggregates = speed_aggregates[speed_aggregates['session'] == 'Race']

    non_numerical_columns = ['seasonyear', 'eventname', 'session', 'driver']
    num_race_columns =  list(filterfalse(lambda x: x in non_numerical_columns, 
                                         list(race_speed_aggregates.columns)))
    for col in num_race_columns:
        race_speed_aggregates = race_speed_aggregates.rename(columns={col: 'race_' + col})
    race_speed_aggregates = race_speed_aggregates.drop(columns=['session'])
    return race_speed_aggregates


def transform_lap_sector_time_columns(session_data, columns):
    """Convert lap and sector time columns from timedelta to seconds.

    Args:
        df (pd.DataFrame): Input dataframe with timedelta columns.

    Returns:
        pd.DataFrame: Dataframe with times in seconds.
    """
    session_data = session_data.copy()
    for col in columns:
        new_col_name_s = str(col + '_s')
        try:
            session_data[new_col_name_s] = session_data[col].dt.total_seconds()
        except AttributeError as e:
            print(e)

    return session_data

def transform_qualifying_data(session_results):
    """Extract relevant qualifying session features.

    Args:
        df (pd.DataFrame): Qualifying session data.

    Returns:
        pd.DataFrame: Processed qualifying features including Q1-Q3 times.
    """
    qualifying = session_results.copy()
    time_columns = ['q1', 'q2', 'q3']
    transformed_qualifying = qualifying.pipe(transform_lap_sector_time_columns, time_columns)
    transformed_qualifying['qtime_s'] = np.where(transformed_qualifying['q2_s'].isna(),
                                                 transformed_qualifying['q1_s'],
                                                 transformed_qualifying['q2_s'])
    transformed_qualifying['qtime_s'] = np.where(transformed_qualifying['q3_s'].isna(), 
                                                 transformed_qualifying['qtime_s'], 
                                                 transformed_qualifying['q3_s'] )
    transformed_qualifying = transformed_qualifying[['seasonyear','eventname','session', 
                                                     'abbreviation', 'q1_s', 
                                                     'q2_s', 'q3_s', 'qtime_s']]
    transformed_qualifying = transformed_qualifying.rename(columns={'abbreviation': 'driver'})
    transformed_qualifying['startingposition'] = transformed_qualifying.groupby(['seasonyear', 
                                                                                 'eventname', 
                                                                                 'session']).cumcount() + 1
    transformed_qualifying = transformed_qualifying.drop(columns=['session'])
    return transformed_qualifying

def transform_race_data(session_results):
    """Transform race results into useful features for modeling.

    Args:
        df (pd.DataFrame): Race results including driver status and positions.

    Returns:
        pd.DataFrame: Dataframe with final position, status flags, and finish rate.
    """
    race = session_results[session_results['session'] == 'Race']
    race = race.copy()
    race['finalposition'] = race.groupby(['seasonyear', 'eventname', 'session']).cumcount() + 1
    race['time_s'] = race['time'].dt.total_seconds()
    race['race_time_s'] = race.groupby(['seasonyear', 'eventname', 'session'])['time_s'].cumsum()
    race['finished'] = np.where(race['status'] == 'Finished', True, False)
    race = race[['seasonyear','eventname','session', 'teamid','abbreviation',
                  'race_time_s','status','finished','finalposition','points']]
    race = race.rename(columns={'abbreviation': 'driver'})
    return race

def get_earliest_latest_session_dateutc(schedule):
    schedule_groupby = schedule.groupby(['seasonyear', 'roundnumber', 'eventname'])\
                        .agg({'sessiondateutc': ['min', 'max']}).reset_index()
    schedule_groupby.columns = ['seasonyear', 'roundnumber', 'eventname',
                                 'practice_1_dateutc', 'race_dateutc']
    return schedule_groupby

def get_driver_standings(race_results, race_schedule):
    """Extract earliest and latest UTC timestamps for each event.

    Args:
        df (pd.DataFrame): Event schedule data.

    Returns:
        pd.DataFrame: Event-level min and max UTC datetimes.
    """
    current_utc_time = pd.to_datetime(datetime.now(timezone.utc))
    current_past_races = race_schedule[race_schedule['practice_1_dateutc'] < current_utc_time]
    race_schedule_results = current_past_races[['seasonyear','roundnumber', 'eventname']]\
                                                .merge(race_results,
                                                        on=['seasonyear', 'eventname'],
                                                        how='left')

    # Ensure correct sorting before cumulative operations
    race_schedule_results = race_schedule_results.sort_values(['seasonyear', 'driver', 'roundnumber'])

    # Group and transform instead of agg to keep full rows
    race_schedule_results['cum_points'] = race_schedule_results\
                                            .groupby(['seasonyear', 'driver'])['points'].cumsum()
    race_schedule_results['cum_points_at_qualifiying_team'] = race_schedule_results['cum_points']\
                                                                 - race_schedule_results['points']
    race_schedule_results['cum_finalposition'] = race_schedule_results\
                                                    .groupby(['seasonyear', 'driver'])['finalposition'].cumsum()
    race_schedule_results['cum_finalposition_at_qualifying_team'] = race_schedule_results['cum_finalposition']\
                                                                         - race_schedule_results['finalposition']
    race_schedule_results['avg_finalposition_at_qualifying_team'] = round(race_schedule_results['cum_finalposition_at_qualifying_team']
                                                                             / race_schedule_results['roundnumber'],2)
    race_schedule_results['in_points_finished'] = np.where(race_schedule_results['finished'] is True, 1,0)
    race_schedule_results['cum_races_in_points_team'] = race_schedule_results.groupby(['seasonyear', 'driver'])['in_points_finished'].cumsum()
    race_schedule_results['cum_races_in_points_at_qualifying_team'] = race_schedule_results['cum_races_in_points_team']\
                                                                         - race_schedule_results['in_points_finished']
    race_schedule_results['avg_races_in_points_at_qualifying_team'] = round(race_schedule_results['cum_races_in_points_at_qualifying_team']
                                                                             / race_schedule_results['roundnumber'], 2)

    race_schedule_results = race_schedule_results[['seasonyear','roundnumber','eventname',
                                                   'teamid','driver', 'cum_points_at_qualifiying_team',
                                                    'avg_finalposition_at_qualifying_team',
                                                    'avg_races_in_points_at_qualifying_team']]
    return race_schedule_results

def get_team_standings(race_results, race_schedule):
    """Compute cumulative driver statistics up to each qualifying event.

    Args:
        df (pd.DataFrame): Driver standings with points and wins.

    Returns:
        pd.DataFrame: Aggregated driver standings per event.
    """
    current_utc_time = pd.to_datetime(datetime.now(timezone.utc))
    current_past_races = race_schedule[race_schedule['practice_1_dateutc'] < current_utc_time]
    race_schedule_results = current_past_races[['seasonyear','roundnumber', 'eventname']]\
                                .merge(race_results, on=['seasonyear', 'eventname'], how='left')

    # Ensure correct sorting before cumulative operations
    race_schedule_results = race_schedule_results[['seasonyear', 'roundnumber', 'eventname', 
                                                   'teamid', 'points', 'finalposition']]
    race_schedule_results = race_schedule_results.sort_values(['seasonyear', 'teamid', 'roundnumber'])

    teams_per_grand_prix = race_schedule_results[['seasonyear','roundnumber','eventname','teamid']]\
                                                    .drop_duplicates()\
                                                    .sort_values(by=['seasonyear', 'teamid', 'roundnumber'])
    points_per_team = race_schedule_results.groupby(['seasonyear', 'roundnumber','teamid'])['points'].sum().reset_index()
    finalposition_per_team = race_schedule_results\
                                .groupby(['seasonyear', 'roundnumber', 'teamid'])['finalposition'].mean().reset_index()
    # Combine Group By Data 
    teams_per_gp_w_points = teams_per_grand_prix.merge(points_per_team, 
                                                       on=['seasonyear', 'roundnumber', 'teamid'], 
                                                       how='left')
    teams_per_gp_w_points_positions = teams_per_gp_w_points.merge(finalposition_per_team, 
                                                                  on=['seasonyear', 'roundnumber', 'teamid'], 
                                                                  how='left')
    # Points per Team 
    teams_per_gp_w_points_positions['cum_points'] = teams_per_gp_w_points_positions.groupby(['seasonyear', 'teamid'])['points'].cumsum()
    teams_per_gp_w_points_positions['cum_points_at_qualifying'] = teams_per_gp_w_points_positions['cum_points']\
                                                                     - teams_per_gp_w_points_positions['points']
    teams_per_gp_w_points_positions['avg_points_at_qualifying'] = round(teams_per_gp_w_points_positions['cum_points_at_qualifying']
                                                                         / teams_per_gp_w_points_positions['roundnumber'],2)
    # Final Position per Team
    teams_per_gp_w_points_positions['cum_finalposition'] = teams_per_gp_w_points_positions\
                                                            .groupby(['seasonyear', 'teamid'])['finalposition'].cumsum()
    teams_per_gp_w_points_positions['cum_finalposition_at_qualifying'] = teams_per_gp_w_points_positions['cum_finalposition']\
                                                                             - teams_per_gp_w_points_positions['finalposition']
    teams_per_gp_w_points_positions['avg_finalposition_at_qualifying'] = round(teams_per_gp_w_points_positions['cum_finalposition_at_qualifying']
                                                                                / teams_per_gp_w_points_positions['roundnumber'],2)

    teams_per_gp_w_points_positions = teams_per_gp_w_points_positions[['seasonyear','roundnumber', 
                                                                       'eventname','teamid', 
                                                                       'avg_points_at_qualifying', 
                                                                       'avg_finalposition_at_qualifying']]
    return teams_per_gp_w_points_positions

def fillna_with_zero(data,null_columns):
    """Fill NaN values with zero in the specified columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): List of column names to fill.

    Returns:
        pd.DataFrame: Updated dataframe with NaNs replaced.
    """
    data = data.copy()
    for column in null_columns:
        #data[column].fillna(0, inplace=True)
        data.fillna({column: 0}, inplace=True)
    return data

def shorten_event_name(event_name):
    """Simplify event names by truncating and removing year information.

    Args:
        name (str): Original event name.

    Returns:
        str: Simplified event name.
    """
    event_name_short = event_name.lower().replace(' ','_').replace('_grand_prix','')
    return event_name_short

def combine_data(practice_data, qualifying_data, 
                 driver_team_standings, race_results, 
                 race_speed_aggregates, nullable_columns):
    """Merge all features into a single dataset per driver per event.

    Args:
        practice_data (pd.DataFrame): Processed practice session data.
        quali_data (pd.DataFrame): Qualifying session features.
        race_data (pd.DataFrame): Final race features and outcomes.
        driver_standings (pd.DataFrame): Cumulative driver stats.
        team_standings (pd.DataFrame): Cumulative team stats.

    Returns:
        pd.DataFrame: Final dataset with all merged features per driver-event.
    """
    race_speed_aggregates = race_speed_aggregates[['seasonyear', 'eventname', 'driver', \
                                                   'race_laptime_s_min', 'race_num_laps', 
                                                   'race_laptime_s_mean']]
    practice_qualifying_data = pd.merge(practice_data,
                                         qualifying_data,
                                         on =['seasonyear', 'eventname', 'driver'],
                                         how='left')
    pract_qual_with_standings = pd.merge(practice_qualifying_data, 
                                         driver_team_standings, 
                                         on =['seasonyear', 'eventname', 'driver'], 
                                         how='left')
    pract_qual_with_standings_race_results = pd.merge(pract_qual_with_standings, 
                                                      race_results, 
                                                      on =['seasonyear', 'eventname', 'teamid','driver'], 
                                                      how='left')
    pract_qual_with_standings_race_results_agg = pd.merge(pract_qual_with_standings_race_results, 
                                                          race_speed_aggregates, 
                                                          on =['seasonyear', 'eventname', 'driver'], 
                                                          how='left')
    pract_qual_with_standings_race_results_agg  = fillna_with_zero(pract_qual_with_standings_race_results_agg, 
                                                                   nullable_columns)
    pract_qual_with_standings_race_results_agg['calc_race_avg_laptime'] = pract_qual_with_standings_race_results_agg['race_time_s'] / pract_qual_with_standings_race_results_agg['race_num_laps']
    pract_qual_with_standings_race_results_agg['eventname_abbr'] = pract_qual_with_standings_race_results_agg['eventname'].apply(shorten_event_name)

    return pract_qual_with_standings_race_results_agg