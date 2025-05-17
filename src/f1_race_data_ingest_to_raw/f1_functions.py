"""Functions for FastF1 data transformations"""
import pandas as pd
import fastf1

def get_schedules(seasons):
    """
    Retrieve event schedules and return them as a pandas DataFrame.

    This function iterates over the specified seasons and retrieves the event schedule 
    for each year using the FastF1 library. The event schedules are collected into a 
    list and then concatenated into a single pandas DataFrame.

    Args:
        seasons (list): A list of seasons (years) for which to retrieve event schedules.

    Returns:
        pandas.DataFrame: A DataFrame containing the event schedules for all specified years.
    """
    schedule_list = []
    for i in seasons:
        event_schedule = fastf1.get_event_schedule(i)
        event_schedule['SeasonYear'] = i
        schedule_list.append(event_schedule)
    return pd.concat(schedule_list)

def transform_schedules(schedules):
    """
    Transform the event schedules DataFrame into a session-based format.

    This function extracts session information from the provided schedules DataFrame,
    renaming columns for clarity and filtering out sessions that are marked as 'None'.
    The resulting DataFrame is sorted by session date in UTC.

    Args:
        schedules (pandas.DataFrame): A DataFrame containing event schedules with session details.

    Returns:
        pandas.DataFrame: A transformed DataFrame containing session schedules, 
                          with columns for session name, date, and other relevant details.
    """
    base_columns = ['SeasonYear', 'RoundNumber', 'Country', 'Location',
                    'OfficialEventName', 'EventDate', 'EventName']
    session_schedules = []
    for i in range(1, 6):
        session_schedule = schedules[base_columns + ['EventFormat', f'Session{i}',
                                                      f'Session{i}Date', f'Session{i}DateUtc']]
        session_schedule = session_schedule.rename(columns={f'Session{i}': 'Session',
                                                            f'Session{i}Date': 'SessionDate', 
                                                            f'Session{i}DateUtc': 'SessionDateUtc'})
        session_schedules.append(session_schedule)

    session_schedules = pd.concat(session_schedules)
    session_schedules = session_schedules[session_schedules['Session'] != 'None']
    session_schedules['SessionDateUtc'] = pd.to_datetime(session_schedules['SessionDateUtc'],
                                                          utc=True)
    session_schedules = session_schedules.sort_values(by=['SessionDateUtc'])
    return session_schedules
