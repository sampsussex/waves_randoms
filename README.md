# WAVES Random Catalog Generator

This repository now uses a single entry point:

- **`make_waves_randoms.py`**: generate randoms for WAVES-wide (`waves-wide_n`, `waves-wide_s`), WAVES-deep (`waves-deep`), and WD hex regions (`WD01`, `WD02`, `WD03`, `WD10`).

## Requirements

- Python 3.8+
- Python packages: `numpy`, `scipy`, `pyarrow`, `regionx`

Install dependencies (example):

```bash
pip install numpy scipy pyarrow regionx
```

## Input Data

Mask files live in the `masks/` directory by default.

- **Aperture masks**: whitespace-delimited files with `ra dec radius` per line (comments allowed with `#`).
- **Polygon masks**: whitespace-delimited polygon vertices (`ra1 dec1 ra2 dec2 ...`).

Default mask files are now selected automatically from `masks/` based on `--region`:

- `waves-wide_n`: `starmask_waves_n.dat`, `ghostmask_waves_n.dat`, `ngc_n.dat`
- `waves-wide_s`: `starmask_waves_s.dat`, `ghostmask_waves_s.dat`, `ngc_s.dat`, `extra_waves_s_sources.dat`
- `waves-deep`: `starmask_waves_s.dat`, `ghostmask_waves_s.dat`, `ngc_s.dat`, `extra_waves_s_sources.dat`
- `WD01/WD02/WD03/WD10`: `WDxx_stars.dat`, `WDxx_ghosts.dat` (no default polygon/extra mask)

> Note: NGC files are used as polygon masks.

## Usage

### WAVES-wide / WAVES-deep

```bash
python make_waves_randoms.py \
  --region waves-wide_n \
  --nrandoms 1000000 \
  --save-location ./waves_randoms/
```

### WD hex region

```bash
python make_waves_randoms.py \
  --region WD01 \
  --nrandoms 1000000 \
  --save-location ./wd_randoms/
```

Skip masks entirely:

```bash
python make_waves_randoms.py --region waves-wide_s --no-masks
```

Use an input parquet catalog instead of generating random coordinates (defaults to `RAmax`/`Decmax`):

```bash
python make_waves_randoms.py \
  --region waves-wide_n \
  --input-catalog-parquet ./input_catalog.parquet \
  --input-ra-column RAmax \
  --input-dec-column Decmax \
  --save-location ./waves_randoms/
```

## Outputs

The script writes chunked Parquet files to `--save-location` and includes:

- `ra`, `dec`
- `in_region`
- `starmask`, `ghostmask`, `polygon_mask`
- `realisation`
