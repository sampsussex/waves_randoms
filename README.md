# WAVES Random Catalog Generator

This repository contains scripts for generating random catalogs for WAVES survey regions. There are two entry points:

- **`make_waves_wide_randoms.py`**: generate randoms for the WAVES-wide North/South regions and the WAVES-deep region.
- **`make_waves_ddf_randoms.py`**: generate randoms for WAVES Deep Field (WD) hex regions.

Both scripts can optionally apply aperture- and polygon-based masks and write results to chunked Parquet files.

## Requirements

- Python 3.8+
- Python packages: `numpy`, `scipy`, `pyarrow`, `regionx`, `tqdm`

Install dependencies (example):

```bash
pip install numpy scipy pyarrow regionx tqdm
```

## Input Data

Mask files live in the `masks/` directory by default. Each script has CLI flags to override mask paths or to skip masks entirely.

- **Aperture masks**: whitespace-delimited files with `ra dec radius_deg` per line (comments allowed with `#`).
- **Polygon masks**: whitespace-delimited polygon vertices, as expected by `regionx`.

## Usage

### WAVES-wide randoms

Generate a WAVES-wide North catalog with 1 million points and default masks:

```bash
python make_waves_wide_randoms.py \
  --region waves-wide_n \
  --nrandoms 1000000 \
  --save-location ./waves_randoms/
```

Skip masks entirely:

```bash
python make_waves_wide_randoms.py \
  --region waves-wide_s \
  --nrandoms 500000 \
  --no-masks
```

### WAVES Deep Field (WD) hex randoms

Generate WD01 randoms with polygon acceptance and masks:

```bash
python make_waves_ddf_randoms.py \
  --region WD01 \
  --nrandoms 1000000 \
  --save-location ./wd_randoms/
```

Run without masks:

```bash
python make_waves_ddf_randoms.py \
  --region WD03 \
  --nrandoms 500000 \
  --no-masks
```

## Outputs

Both scripts write chunked Parquet files to the `--save-location` directory. Output tables include:

- `ra`, `dec`: sky coordinates in degrees.
- `starmask`, `ghostmask`, `polygon_mask`: boolean mask flags.
- `realisation`: integer flag for block realisations.

## Notes

- Right Ascension (RA) values are normalized to `[0, 360)` degrees.
- Declination (Dec) randoms are sampled uniformly in area.
- For large runs, adjust `--chunk-size` to manage memory usage.
