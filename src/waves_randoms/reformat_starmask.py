import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

gaia_stars_filepath = '/Users/sp624AA/Downloads/gaiastarmaskwaves.csv'
output_dir = '/Users/sp624AA/Downloads/repacked_starmask_data'
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Masking radius rules
# ---------------------------------------------------------------------------

def mask_radius_gama_ddf_rule(g):
    """
    Calculate r[deg] based on the GAMA/DDF rule.

    Parameters:
        g (float or array-like): Input magnitude.

    Returns:
        numpy.ndarray: Calculated r[deg].
    """
    g = np.asarray(g)  # Ensure g is a NumPy array
    r = np.zeros_like(g, dtype=float)
    mask1 = (g < 16)
    r[mask1] = (10 ** (1.6 - 0.15 * g[mask1]))

    mask2 = r > 5
    r[mask2] = 5
    return r / 60


def mask_radius_wwns_wd_rule(g):
    """
    Calculate r[deg] based on the WAVES-Wide N/S rule.

    Parameters:
        g (float or array-like): Input magnitude.

    Returns:
        numpy.ndarray: Calculated r[deg].
    """
    g = np.asarray(g)  # Ensure g is a NumPy array
    r = np.zeros_like(g, dtype=float)
    mask1 = (g > 3.5) & (g < 16)
    mask2 = g <= 3.5
    r[mask1] = (10 ** (1.3 - 0.13 * g[mask1]))
    r[mask2] = 7
    return r / 60


# ---------------------------------------------------------------------------
# Region definitions
#
# WW-N and WW-S boundaries are padded by 0.5 deg beyond the quoted survey
# footprint (per field). WD field boundaries are instead fixed at exactly
# +/-3 deg from the field centre, as specified.
#
# 'wrap' = True means the RA range crosses the 0/360 boundary, so the
# selection is ra >= ra_min OR ra <= ra_max (rather than an AND).
# ---------------------------------------------------------------------------

regions = [
    {
        'name': 'WAVES-Wide N',
        'code': 'WWN',
        'ra_min': 157.25 - 0.5,
        'ra_max': 225.0 + 0.5,
        'dec_min': -3.95 - 0.5,
        'dec_max': 3.95 + 0.5,
        'wrap': False,
        'rule': mask_radius_wwns_wd_rule,
    },
    {
        'name': 'WAVES-Wide S',
        'code': 'WWS',
        'ra_min': 330.0 - 0.5,
        'ra_max': 51.6 + 0.5,
        'dec_min': -35.6 - 0.5,
        'dec_max': -27.0 + 0.5,
        'wrap': True,
        'rule': mask_radius_wwns_wd_rule,
    },
    {
        'name': 'WD01',
        'code': 'WD01',
        'ra_min': 9.50 - 3.0,
        'ra_max': 9.50 + 3.0,
        'dec_min': -43.95 - 3.0,
        'dec_max': -43.95 + 3.0,
        'wrap': False,
        'rule': mask_radius_gama_ddf_rule,
    },
    {
        'name': 'WD02',
        'code': 'WD02',
        'ra_min': 35.875 - 3.0,
        'ra_max': 35.875 + 3.0,
        'dec_min': -5.025 - 3.0,
        'dec_max': -5.025 + 3.0,
        'wrap': False,
        'rule': mask_radius_gama_ddf_rule,
    },
    {
        'name': 'WD03',
        'code': 'WD03',
        'ra_min': 53.125 - 3.0,
        'ra_max': 53.125 + 3.0,
        'dec_min': -28.1 - 3.0,
        'dec_max': -28.1 + 3.0,
        'wrap': False,
        'rule': mask_radius_gama_ddf_rule,
    },
    {
        'name': 'WD10',
        'code': 'WD10',
        'ra_min': 150.125 - 3.0,
        'ra_max': 150.125 + 3.0,
        'dec_min': 2.20 - 3.0,
        'dec_max': 2.20 + 3.0,
        'wrap': False,
        'rule': mask_radius_gama_ddf_rule,
    },
]

# Note: WAVES-Deep 23 (WD-23) is intentionally excluded.

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

stars = pd.read_csv(gaia_stars_filepath)

# ---------------------------------------------------------------------------
# Process each region and save a .dat file
# ---------------------------------------------------------------------------
print(stars.head())

for region in regions:
    ra = stars['ra']
    dec = stars['dec']

    if region['wrap']:
        ra_mask = (ra >= region['ra_min']) | (ra <= region['ra_max'])
    else:
        ra_mask = (ra >= region['ra_min']) & (ra <= region['ra_max'])

    dec_mask = (dec >= region['dec_min']) & (dec <= region['dec_max'])

    # Drop stars too faint to produce a meaningful mask (g >= 16), matching
    # the original selection logic.
    mag_mask = stars['phot_g_mean_mag'] < 16

    subset = stars[ra_mask & dec_mask & mag_mask].copy()

    subset['masking_radii[deg]'] = region['rule'](subset['phot_g_mean_mag'])

    subset = subset[['ra', 'dec', 'masking_radii[deg]']]

    out_path = os.path.join(output_dir, f"{region['code']}_stars.dat")
    subset.to_csv(out_path, sep=' ', index=False, header=False)

    print(f"{region['name']} ({region['code']}): {len(subset)} stars -> {out_path}")