
from __future__ import annotations

import collections.abc
import contextlib
import functools
from typing import Iterator, TypeVar
import typing

import npc_lims
import pandas as pd
import polars as pl

def get_dfs(version: str | None = None, with_bool_columns: bool = True) -> typing.Mapping[str | npc_lims.NWBComponentStr, pd.DataFrame]:
    """Get a dictionary of dataframes for each table-like component in an NWB file (except units)."""
    components = (c for c in typing.get_args(npc_lims.NWBComponentStr) if c != "units")
    def _helper(component, version, with_bool_columns) -> pd.DataFrame:
        df = pd.read_parquet(npc_lims.get_cache_path(component, version=version))
        if with_bool_columns:
            df = add_bool_columns(df)
        return df
    return LazyDict({
        component: (_helper, (component, version, with_bool_columns), {})
        for component in components
    })

def add_epoch_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to add name column to epochs row of dataframe
    Assumes each row has subject_id, date, and session idx columns

    >>> df = pd.DataFrame({'tags': [['DynamicRouting1']]})
    >>> df        
    0               tags
    0  [DynamicRouting1]
    >>> add_epoch_name_column(df)
    0               name    
    0   DynamicRouting1
    """
    tags = df.copy().explode('tags')
    is_epoch_name = tags.tags.str.contains('[A-Z]', regex=True)
    names = tags[is_epoch_name]
    names = names.rename(columns={'tags': 'name'})
    return names

@functools.cache
def get_session_bools_df(version: str | None = None) -> pd.DataFrame:
    """Get a dataframe with session_id, is_ephys, is_templeton, is_training, is_dynamic_routing columns.
    
    >>> get_session_bools_df().columns
    Index(['session_id', 'is_ephys', 'is_templeton', 'is_training',
           'is_dynamic_routing', 'is_opto'],
          dtype='object')
    """
    session_df = pd.read_parquet(npc_lims.get_cache_path('session', version=version))
    session_df = add_session_id_column(session_df)
    bools_df = pd.DataFrame(
        dict(
            is_ephys = session_df.keywords.map({'ephys'}.issubset),
            is_templeton = (is_templeton := session_df.keywords.map({'Templeton'}.issubset)),
            is_training = session_df.keywords.map({'training'}.issubset),
            is_dynamic_routing = ~is_templeton,
            is_opto = session_df.keywords.map({'opto'}.issubset),
        )
    )
    session_bools_df = pd.concat([session_df['session_id'].reset_index(drop=True), bools_df.reset_index(drop=True)], axis=1)
    return session_bools_df

def add_is_task_first_column(epochs: pd.DataFrame):
    epochs['is_task_first'] = None
    for session_id in epochs.session_id.unique():
        e = epochs[epochs.session_id == session_id]
        has_task = 'DynamicRouting1' in e.name.values
        if not has_task:
            continue
        is_task_first = e[e.name == 'DynamicRouting1'].start_time.values[0] == e.start_time.min()
        epochs.loc[epochs.session_id == session_id, 'is_task_first'] = is_task_first
    return epochs
    
def add_session_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    >>> df = pd.DataFrame({'subject_id': ['660023'], 'date': ['2023-08-09'], 'session_idx': [0]})
    >>> df
        subject_id        date  session_idx
    0       660023  2023-08-09            0
    >>> add_session_id_column(df)
        subject_id        date  session_idx           session_id
    0       660023  2023-08-09            0  660023_2023-08-09_0
    """
    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        df_copy['session_id'] = df.apply(lambda row: f"{row['subject_id']}_{row['date']}_{row['session_idx']}", axis=1)   
        return df_copy
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df.with_columns(
            pl.concat_str('subject_id', 'date', 'session_idx', separator = '_')
            .alias('session_id')
        )
    
def add_bool_columns(df: pd.DataFrame, version: str | None = None) -> pd.DataFrame:
    """
    Function to add bool columns: is_ephys, is_templeton, is_training as columns to row of dataframe
    Assumes each row has subject_id, date, and session idx columns

    >>> df = pd.DataFrame({'subject_id': ['660023'], 'date': ['2023-08-09'], 'session_idx': [0]})
    >>> df
        subject_id        date  session_idx
    0       660023  2023-08-09            0
    >>> add_bool_columns(df)
        subject_id        date  session_idx           session_id  is_ephys  is_templeton  is_training  is_dynamic_routing  is_opto
    0       660023  2023-08-09            0  660023_2023-08-09_0      True         False        False                True    False
    """
    session_bools_df = get_session_bools_df(version=version)
    return add_session_id_column(df).merge(session_bools_df, on=['session_id'])  

K = TypeVar("K")
V = TypeVar("V")


class LazyDict(collections.abc.Mapping[K, V]):
    """Dict for postponed evaluation of functions and caching of results.

    Assign values as a tuple of (callable, args, kwargs). The callable will be
    evaluated when the key is first accessed. The result will be cached and
    returned directly on subsequent access.

    Effectively immutable after initialization.

    Initialize with a dict:
    >>> d = LazyDict({'a': (lambda x: x + 1, (1,), {})})
    >>> d['a']
    2

    or with keyword arguments:
    >>> d = LazyDict(b=(min, (1, 2), {}))
    >>> d['b']
    1
    """

    def __init__(self, *args, **kwargs) -> None:
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key) -> V:
        with contextlib.suppress(TypeError):
            func, args, *kwargs = self._raw_dict.__getitem__(key)
            self._raw_dict.__setitem__(key, func(*args, **kwargs[0]))
        return self._raw_dict.__getitem__(key)

    def __iter__(self) -> Iterator[K]:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={list(self._raw_dict.keys())})"


if __name__ == '__main__':
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
    
