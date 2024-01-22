# dynamic-routing-retreat-2024-01

## quickstart

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