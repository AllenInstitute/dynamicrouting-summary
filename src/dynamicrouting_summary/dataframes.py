from __future__ import annotations

import typing

import npc_lims
import pandas as pd
import dynamicrouting_summary.utils as utils


def get_dfs(version: str | None = None, with_bool_columns: bool = True) -> dict[str | npc_lims.NWBComponentStr, pd.DataFrame]:
    """Get a dictionary of dataframes for each table-like component in an NWB file (except units)."""
    components = (c for c in typing.get_args(npc_lims.NWBComponentStr) if c != "units")
    dfs = {
        component: pd.read_parquet(npc_lims.get_cache_path(component, version=version))
        for component in components
    }
    if with_bool_columns:
        return {k: utils.add_bool_columns(v) for k, v in dfs.items()}
    else:
        return dfs