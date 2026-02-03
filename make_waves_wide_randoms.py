#!/usr/bin/env python3
import os
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
from regionx import Aperture
from regionx import Polygon as regionx_polygon  # avoid name clash
from scipy.spatial import cKDTree


import pyarrow as pa
import pyarrow.parquet as pq


# Notes:
# - regionx expects RA/Dec in degrees.
# - This script normalizes all RA to [0, 360).
# - WAVES-S crosses RA=0; we handle wrapped RA ranges properly.
# - For large nrandoms, keep chunk_size sensible to avoid RAM blow-ups.


def normalize_ra(ra_deg: np.ndarray) -> np.ndarray:
    """Map RA to [0, 360)."""
    return np.mod(ra_deg, 360.0)


def sample_ra_wrapped(rng: np.random.Generator, ramin: float, ramax: float, n: int) -> np.ndarray:
    """
    Sample RA uniformly from an interval that may wrap around 0/360.
    Inputs can be in any convention (e.g., negative RA); output is [0, 360).
    Examples:
      ramin=-30, ramax=51.6   -> interval [330, 360) U [0, 51.6]
      ramin=157.25, ramax=225 -> normal interval
    """
    ramin_n = float(ramin) % 360.0
    ramax_n = float(ramax) % 360.0

    if np.isclose(ramin_n, ramax_n):
        raise ValueError(f"RA range has ~zero width after normalization: ramin={ramin}, ramax={ramax}")

    if ramin_n < ramax_n:
        return rng.uniform(ramin_n, ramax_n, size=n)

    # Wrapped interval
    width = (360.0 - ramin_n) + ramax_n
    u = rng.random(size=n) * width
    ra = ramin_n + u
    return normalize_ra(ra)


def sample_dec_equal_area(rng: np.random.Generator, decmin: float, decmax: float, n: int) -> np.ndarray:
    """Sample Dec uniformly in *area* within [decmin, decmax] (degrees)."""
    if decmin >= decmax:
        raise ValueError(f"Invalid Dec limits: decmin={decmin} >= decmax={decmax}")
    if decmin < -90.0 or decmax > 90.0:
        raise ValueError(f"Dec limits out of bounds [-90, 90]: decmin={decmin}, decmax={decmax}")

    smin = np.sin(np.deg2rad(decmin))
    smax = np.sin(np.deg2rad(decmax))
    u = rng.uniform(smin, smax, size=n)
    return np.rad2deg(np.arcsin(u))


def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """
    Convert RA/Dec in degrees to 3D unit vectors (x,y,z).
    Returns shape (N, 3) float64.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z)).astype(np.float64, copy=False)


def polygon_ball_radius_chord(ra_vertices: List[float], dec_vertices: List[float]) -> Tuple[np.ndarray, float]:
    """
    Return the unit-vector for the first vertex and the maximum chord distance
    from that vertex to all other vertices.
    """
    ra_arr = np.asarray(ra_vertices, dtype=np.float64)
    dec_arr = np.asarray(dec_vertices, dtype=np.float64)
    vertices_xyz = radec_to_unitvec(ra_arr, dec_arr)
    first_xyz = vertices_xyz[0]
    max_radius = np.linalg.norm(vertices_xyz - first_xyz, axis=1).max()
    return first_xyz, float(max_radius)



@dataclass
class ApertureMaskRow:
    ra: float
    dec: float
    radius_deg: float


@dataclass
class PolygonMaskRow:
    ra_vertices: List[float]
    dec_vertices: List[float]


class waveswiderandoms:
    def __init__(
        self,
        seed: int = 42,
        nrandoms: int = 50_000_000,
        starmask_path: str = "./masks/ghostmask_waves_n.dat", #
        ghostmask_path: str = "./masks/ghostmas_waves_n.dat", #
        polygonmask_path: str = "./masks/ngc_n.dat", #
        extra_waves_s_path: str = "./extra_waves_s_sources.dat",
        save_location: str = "./waves_randoms/",
        chunk_size: int = 1_000_000,
    ):
        self.rng = np.random.default_rng(seed)
        self.nrandoms = int(nrandoms)
        self.chunk_size = int(chunk_size)

        waves_n_limits = {"ramax": 225.0, "decmax": 3.95, "ramin": 157.25, "decmin": -3.95}
        waves_s_limits = {"ramax": 51.6, "decmax": -27.0, "ramin": -30.0, "decmin": -35.6}
        waves_deep_limits = {"ramax": 351, "decmax": -30 ,"ramin": 339, "decmin":-35}

        self.regions: Dict[str, Dict[str, float]] = {
            "waves-wide_n": waves_n_limits,
            "waves-wide_s": waves_s_limits,
            "waves-deep": waves_deep_limits,
        }

        self.region_randoms = {
            region: {"ra": None, "dec": None, "starmask": None, "ghostmask": None, "polygon_mask": None, "realisation": None}
            for region in self.regions.keys()
        }

        # Mask catalogs (loaded into memory)
        self.star_mask_catalog: List[ApertureMaskRow] = []
        self.ghostmask_catalog: List[ApertureMaskRow] = []
        self.polygon_catalog: List[PolygonMaskRow] = []
        self.waves_s_extra_catalog: List[ApertureMaskRow] = []

        self.starmask_path = starmask_path
        self.ghostmask_path = ghostmask_path
        self.polygonmask_path = polygonmask_path
        self.extra_waves_s_path = extra_waves_s_path
        self.save_location = save_location

        os.makedirs(self.save_location, exist_ok=True)

    # -----------------------------
    # Random generation
    # -----------------------------
    def generate_randoms(self, region: str, nrandoms: Optional[int] = None) -> None:
        """
        Generates random points in the specified region and stores them in self.region_randoms[region].
        """
        if region not in self.regions:
            raise KeyError(f"Unknown region '{region}'. Known: {list(self.regions.keys())}")

        n = self.nrandoms if nrandoms is None else int(nrandoms)
        lim = self.regions[region]

        ra = sample_ra_wrapped(self.rng, lim["ramin"], lim["ramax"], n)
        dec = sample_dec_equal_area(self.rng, lim["decmin"], lim["decmax"], n)

        self.region_randoms[region]["ra"] = ra
        self.region_randoms[region]["dec"] = dec

        # Initialise mask columns (False means "not masked")
        self.region_randoms[region]["starmask"] = np.zeros(n, dtype=bool)
        self.region_randoms[region]["ghostmask"] = np.zeros(n, dtype=bool)
        self.region_randoms[region]["polygon_mask"] = np.zeros(n, dtype=bool)
        self.region_randoms[region]["realisation"] = np.zeros(n, dtype=np.int32)

    # -----------------------------
    # Mask loading
    # -----------------------------
    def _load_aperture_mask_csv(self, path: str) -> List[ApertureMaskRow]:
        """
        Loads a whitespace-separated file with 3 columns per line:
            ra  dec  radius_deg
        No header.
        Lines starting with '#' are ignored.
        """
        rows: List[ApertureMaskRow] = []
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        bad_lines = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f, start=1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.split()  # splits on any whitespace
                if len(parts) < 3:
                    bad_lines += 1
                    continue

                try:
                    ra = float(parts[0])
                    dec = float(parts[1])
                    rad = float(parts[2])
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno} could not parse floats: {s}") from e

                rows.append(ApertureMaskRow(ra=float(ra) % 360.0, dec=float(dec), radius_deg=float(rad)))

        if len(rows) == 0:
            raise RuntimeError(
                f"Loaded 0 apertures from {path}. "
                f"Check delimiter/format. (bad_lines={bad_lines})"
            )

        print(f"Loaded {len(rows)} apertures from {path} (skipped {bad_lines} short/blank lines)")
        return rows


    def _load_polygon_mask_csv(self, path: str) -> List[PolygonMaskRow]:
        """
        Loads whitespace-separated polygons, one polygon per line:
            ra1 dec1 ra2 dec2 ... raN decN
        No header. N can vary by line, but must be >=3 vertices (>=6 numbers) and even count.
        Lines starting with '#' are ignored.
        """
        polys: List[PolygonMaskRow] = []
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        bad_lines = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f, start=1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.split()
                if len(parts) < 6 or (len(parts) % 2) != 0:
                    bad_lines += 1
                    continue

                try:
                    vals = [float(x) for x in parts]
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno} could not parse floats: {s}") from e

                ra_vertices = [vals[i] % 360.0 for i in range(0, len(vals), 2)]
                dec_vertices = [vals[i] for i in range(1, len(vals), 2)]
                polys.append(PolygonMaskRow(ra_vertices=ra_vertices, dec_vertices=dec_vertices))

        if len(polys) == 0:
            raise RuntimeError(
                f"Loaded 0 polygons from {path}. "
                f"Check delimiter/format. (bad_lines={bad_lines})"
            )

        print(f"Loaded {len(polys)} polygons from {path} (skipped {bad_lines} malformed lines)")
        return polys

    def load_masks(self) -> None:
        """
        Loads star, ghost and polygon masks from files.
        Star/Ghost format: ra, dec, radius_deg
        Polygon format: ra1, dec1, ra2, dec2, ..., raN, decN
        """
        self.star_mask_catalog = self._load_aperture_mask_csv(self.starmask_path)
        self.ghostmask_catalog = self._load_aperture_mask_csv(self.ghostmask_path)
        self.polygon_catalog = self._load_polygon_mask_csv(self.polygonmask_path)
        # Extra WAVES-S masks
        self.waves_s_extra_catalog = self._load_aperture_mask_csv(self.extra_waves_s_path)

    # -----------------------------
    # Mask application
    # -----------------------------
    def _apply_aperture_catalog_kdtree(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        catalog: List[ApertureMaskRow],
        label: str = "aperture",
    ) -> np.ndarray:
        """
        Fast spherical aperture masking using a KD-tree in 3D unit-vector space.

        Returns boolean array: True where point is inside ANY aperture.
        """
        n = ra.size
        out = np.zeros(n, dtype=bool)

        if len(catalog) == 0:
            return out

        # Precompute aperture centres as unit vectors + chord radii
        ap_ra = np.array([row.ra for row in catalog], dtype=np.float64)
        ap_dec = np.array([row.dec for row in catalog], dtype=np.float64)
        ap_rad_deg = np.array([row.radius_deg for row in catalog], dtype=np.float64)

        ap_xyz = radec_to_unitvec(ap_ra, ap_dec)  # (M, 3)
        theta = np.deg2rad(ap_rad_deg)
        ap_r = 2.0 * np.sin(theta / 2.0)          # chord distances

        # Chunk randoms to keep memory/CPU reasonable
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)

            ra_chunk = ra[start:end]
            dec_chunk = dec[start:end]

            xyz = radec_to_unitvec(ra_chunk, dec_chunk)  # (Nc, 3)
            tree = cKDTree(xyz, leafsize=64)

            chunk_mask = np.zeros(end - start, dtype=bool)

            # Loop apertures; query_ball_point is fast in C
            for i in range(ap_xyz.shape[0]):
                # Skip if everything already masked in this chunk
                if chunk_mask.all():
                    break

                idx = tree.query_ball_point(ap_xyz[i], r=ap_r[i])
                if idx:  # list can be empty
                    chunk_mask[np.asarray(idx, dtype=np.int64)] = True

            out[start:end] = chunk_mask

        return out


    def apply_starmask(self, region: str) -> None:
        ra = self.region_randoms[region]["ra"]
        dec = self.region_randoms[region]["dec"]
        if ra is None or dec is None:
            raise RuntimeError(f"No randoms generated for region '{region}'")

        self.region_randoms[region]["starmask"] = self._apply_aperture_catalog_kdtree(
            ra=ra, dec=dec, catalog=self.star_mask_catalog, label="star"
        )


    def apply_ghostmask(self, region: str) -> None:
        ra = self.region_randoms[region]["ra"]
        dec = self.region_randoms[region]["dec"]
        if ra is None or dec is None:
            raise RuntimeError(f"No randoms generated for region '{region}'")

        self.region_randoms[region]["ghostmask"] = self._apply_aperture_catalog_kdtree(
            ra=ra, dec=dec, catalog=self.ghostmask_catalog, label="ghost"
        )


    def apply_polygonmask(self, region: str) -> None:
        """
        Applies polygon masks; fills boolean column polygon_mask (True means masked).
        """
        ra = self.region_randoms[region]["ra"]
        dec = self.region_randoms[region]["dec"]
        if ra is None or dec is None:
            raise RuntimeError(f"No randoms generated for region '{region}'")

        n = ra.size
        out = np.zeros(n, dtype=bool)

        if len(self.polygon_catalog) == 0:
            self.region_randoms[region]["polygon_mask"] = out
            return

        xyz = radec_to_unitvec(ra, dec)
        tree = cKDTree(xyz, leafsize=64)

        for prow in self.polygon_catalog:
            ra_vertices = [v % 360.0 for v in prow.ra_vertices]
            dec_vertices = prow.dec_vertices
            poly = regionx_polygon(ra_vertices, dec_vertices)

            first_xyz, radius = polygon_ball_radius_chord(ra_vertices, dec_vertices)
            idx = tree.query_ball_point(first_xyz, r=radius)
            if not idx:
                continue

            idx_arr = np.asarray(idx, dtype=np.int64)
            ra_list = ra[idx_arr].tolist()
            dec_list = dec[idx_arr].tolist()

            inside = np.asarray(poly.check_points(ra_list, dec_list), dtype=bool)
            if inside.any():
                out[idx_arr[inside]] = True
            if out.all():
                break

        if region == "waves-wide_s":
            # Also apply extra WAVES-S masks
            extra_mask = self._apply_aperture_catalog_kdtree(
                ra=ra, dec=dec, catalog=self.waves_s_extra_catalog, label="waves-s-extra"
            )
            out |= extra_mask


        self.region_randoms[region]["polygon_mask"] = out

    # -----------------------------
    # Extra columns + save
    # -----------------------------
    def add_realisation_flag(self, region: str, block_size: int = 10_000_000) -> None:
        """
        Adds a realisation flag column. Starts at 0 and increases by 1 every block_size entries.
        """
        ra = self.region_randoms[region]["ra"]
        if ra is None:
            raise RuntimeError(f"No randoms generated for region '{region}'")

        n = ra.size
        real = (np.arange(n, dtype=np.int64) // int(block_size)).astype(np.int32)
        self.region_randoms[region]["realisation"] = real

    def save_randoms_parquet_chunked(self, region: str) -> str:
        data = self.region_randoms[region]
        ra = data["ra"]
        dec = data["dec"]
        if ra is None or dec is None:
            raise RuntimeError(f"No randoms generated for region '{region}'")

        outpath = os.path.join(self.save_location, f"{region}_randoms.parquet")

        schema = pa.schema([
            ("ra", pa.float64()),
            ("dec", pa.float64()),
            ("starmask", pa.bool_()),
            ("ghostmask", pa.bool_()),
            ("polygon_mask", pa.bool_()),
            ("realisation", pa.int32()),
        ])

        writer = pq.ParquetWriter(outpath, schema=schema, compression="zstd")

        try:
            n = ra.size
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)

                batch = pa.table(
                    {
                        "ra": ra[start:end].astype(np.float64),
                        "dec": dec[start:end].astype(np.float64),
                        "starmask": data["starmask"][start:end].astype(bool),
                        "ghostmask": data["ghostmask"][start:end].astype(bool),
                        "polygon_mask": data["polygon_mask"][start:end].astype(bool),
                        "realisation": data["realisation"][start:end].astype(np.int32),
                    },
                    schema=schema,
                )
                writer.write_table(batch)
        finally:
            writer.close()

        return outpath



def main():
    parser = argparse.ArgumentParser(description="Generate WAVES randoms and apply regionx masks.")
    parser.add_argument("--region", default="waves-wide_n", choices=["waves-wide_n", "waves-wide_s", "waves-deep"])
    parser.add_argument("--nrandoms", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--save-location", type=str, default="./waves_randoms/")

    # Masks
    parser.add_argument("--starmask-path", type=str, default="./masks/starmask_waves_s.dat") #
    parser.add_argument("--ghostmask-path", type=str, default="./masks/ghostmask_waves_s.dat") #
    parser.add_argument("--polygonmask-path", type=str, default="./masks/ngc_n.dat") #
    parser.add_argument("--extra-waves-s-masks", type=str, default="./masks/extra_waves_s_sources.dat")

    parser.add_argument("--no-masks", action="store_true", help="Skip loading/applying masks")
    parser.add_argument("--no-star", action="store_true", help="Skip applying star mask")
    parser.add_argument("--no-ghost", action="store_true", help="Skip applying ghost mask")
    parser.add_argument("--no-polygon", action="store_true", help="Skip applying polygon mask")

    parser.add_argument("--block-size", type=int, default=10_000_000, help="Realisation block size (default 10M)")

    args = parser.parse_args()

    gen = waveswiderandoms(
        seed=args.seed,
        nrandoms=args.nrandoms,
        starmask_path=args.starmask_path,
        ghostmask_path=args.ghostmask_path,
        polygonmask_path=args.polygonmask_path,
        extra_waves_s_path=args.extra_waves_s_masks,
        save_location=args.save_location,
        chunk_size=args.chunk_size,
    )

    # Decide which regions to run

    regions = [args.region]


    if not args.no_masks:
        gen.load_masks()

    for region in regions:
        print(f"[{region}] generating {args.nrandoms} randoms")
        gen.generate_randoms(region)

        print(f"[{region}] adding realisation flag (block_size={args.block_size})")
        gen.add_realisation_flag(region, block_size=args.block_size)

        if not args.no_masks:
            if not args.no_star:
                print(f"[{region}] applying star mask (n_apertures={len(gen.star_mask_catalog)})")
                gen.apply_starmask(region)
            if not args.no_ghost:
                print(f"[{region}] applying ghost mask (n_apertures={len(gen.ghostmask_catalog)})")
                gen.apply_ghostmask(region)
            if not args.no_polygon:
                print(f"[{region}] applying polygon mask (n_polygons={len(gen.polygon_catalog)})")
                gen.apply_polygonmask(region)

        outpath = gen.save_randoms_parquet_chunked(region)
        print(f"[{region}] saved -> {outpath}")

        # quick summary
        sm = gen.region_randoms[region]["starmask"]
        gm = gen.region_randoms[region]["ghostmask"]
        pm = gen.region_randoms[region]["polygon_mask"]
        if sm is not None:
            print(f"[{region}] masked fractions: star={sm.mean():.4f}, ghost={gm.mean():.4f}, poly={pm.mean():.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
