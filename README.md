# dynamic-routing-retreat-2024-01

## Installation

```bash
git clone https://github.com/AllenInstitute/dynamicrouting-summary
cd dynamicrouting-summary
```

### install with PDM (requires a system Python>=3.9)
```bash
python -m install pipx
pipx run pdm install
```

### install with Conda
```bash
conda create -n DR-summary python>=3.9
conda activate DR-summary
python -m install pdm
pdm install
```

## Manage dependencies
- add dependency: `pdm add numpy`
  - add dev dependency: `pdm add -G dev numpy`
- remove dependency correctly: `pdm remove numpy`
- always commit `pdm.lock` for a reproducible dev environment

## Get dataframes
- first, get .env file with necessary credentials
  [here](https://alleninstitute.sharepoint.com/sites/DynamicRouting/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FDynamicRouting%2FShared%20Documents%2FMeetings%2FDR%20Retreats%2FJune%202023%20Onsite%20Retreat%2F%2Eenv&parent=%2Fsites%2FDynamicRouting%2FShared%20Documents%2FMeetings%2FDR%20Retreats%2FJune%202023%20Onsite%20Retreat&p=true&ga=1)
and copy into the root of the repo

- get available dataframes (except units):

```python
import pandas as pd

import dynamicrouting_summary as dr

dfs: dict[str, pd.DataFrame] = dr.get_dfs()
```

### Get units

- scan virtual dataset of multiple parquet files (one per session), using
  `pyarrow` or `dask` (note: dask cannot handle files with different schemas)

```python
import pyarrow.dataset as ds
import pyarrow.compute as pc
import npc_lims

import dynamicrouting_summary as dr

units = ds.Dataset(npc_lims.get_cache_path('units')) 


pc.value_counts(ds.dataset(d).to_table(['structure'])['structure']).to_pandas()
df = dr.add_bool_columns(df)


# example of push-down filtering
expr = (ds.field('structure') == 'DG') & (ds.field('num_spikes') >  100_000)
df = ds.dataset(d).filter(expr).to_table(columns=['spike_times']).to_pandas()

```