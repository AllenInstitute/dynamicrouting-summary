from __future__ import annotations

import typing

import npc_lims
import pandas as pd


def get_dfs(version: str | None = '0.0.172') -> dict[str | npc_lims.NWBComponentStr, pd.DataFrame]:
    """Get a dictionary of dataframes for each table-like component in an NWB file (except units)."""
    components = (c for c in typing.get_args(npc_lims.NWBComponentStr) if c != "units")
    return {
        component: pd.read_parquet(npc_lims.get_cache_path(component, version=version))
        for component in components
    }
