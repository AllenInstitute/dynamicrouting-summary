import functools

import npc_lims
import pandas as pd


@functools.cache
def get_session_bools_df(version: str | None = None) -> pd.DataFrame:
    """Get a dataframe with session_id, is_ephys, is_templeton, is_training, is_dynamic_routing columns."""
    session_df = pd.read_parquet(npc_lims.get_cache_path('session', version=version))
    session_ids = session_df.apply(lambda row: f"{row['subject_id']}_{row['date']}_{row['session_idx']}", axis=1)
    bools_df = pd.DataFrame([get_bools(s) for s in session_ids])
    session_bools_df = pd.concat([session_df.reset_index(drop=True), bools_df], axis=1,)
    return session_bools_df

def add_bool_columns(df: pd.DataFrame, version: str | None = None) -> pd.DataFrame:
    """
    Function to add bool columns: is_ephys, is_templeton, is_training as columns to row of dataframe
    Assumes each row has subject_id, date, and session idx columns

    >>> dummy_df = pd.DataFrame({'subject_id': ['660023'], 'date': ['2023-08-09'], 'session_idx': [0]})
    >>> add_bool_columns(dummy_df)
      subject_id        date  session_idx  is_ephys  is_templeton  is_training  is_dynamic_routing
    0     660023  2023-08-09            0      True         False        False                True
    """
    return df.merge(get_session_bools_df(version=version), on=['subject_id', 'date', 'session_idx'], how='outer')

@functools.cache
def get_ephys_session_ids() -> set[str]:
    """Get a list of session_ids for ephys sessions."""
    return {info.id for info in npc_lims.get_session_info(is_ephys=True)}

@functools.cache
def get_templeton_session_ids() -> set[str]:
    """Get a list of session_ids for ephys sessions."""
    return {info.id for info in npc_lims.get_session_info(is_templeton=True)}

@functools.cache
def get_bools(session_id: str) -> dict[str, bool]:
    """Get a dict of bools for is_ephys, is_templeton, is_training,
    is_dynamic_routing, etc. for adding as dataframe columns."""
    return {
        "is_ephys": (is_ephys := session_id in tuple(get_ephys_session_ids())),
        "is_templeton": (is_templeton := session_id in tuple(get_templeton_session_ids())),
        "is_training": not is_ephys,
        "is_dynamic_routing": not is_templeton,
    }


if __name__ == '__main__':
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
    
