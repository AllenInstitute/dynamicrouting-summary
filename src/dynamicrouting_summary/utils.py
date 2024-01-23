import npc_lims
import pandas as pd

def add_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to add bool columns: is_ephys, is_templeton, is_training as columns to row of dataframe
    Assumes each row has subject_id, date, and session idx columns

    >>> dummy_df = pd.DataFrame({'subject_id': ['660023'], 'date': ['2023-08-09'], 'session_idx': [0]})
    >>> add_bool_columns(dummy_df)
      subject_id        date  session_idx  is_ephys  is_templeton  is_training  is_dynamic_routing
    0     660023  2023-08-09            0      True         False        False                True
    """

    is_ephys_column: list[bool] = []
    is_templeton_column: list[bool] = []
    is_training_column: list[bool] = []
    is_dynamic_routing_column: list[bool] = []

    for index, row in df.iterrows():
        session = npc_lims.get_session_info(session=f"{row['subject_id']}_{row['date']}_{row['session_idx']}")
        is_ephys_column.append(session.is_ephys)
        is_templeton_column.append(session.is_templeton)
        is_training_column.append(not session.is_ephys)
        is_dynamic_routing_column.append(not session.is_templeton)

    
    df['is_ephys'] = is_ephys_column
    df['is_templeton'] = is_templeton_column
    df['is_training'] = is_training_column
    df['is_dynamic_routing'] = is_dynamic_routing_column

    return df

if __name__ == '__main__':
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
    
