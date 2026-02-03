#!/usr/bin/env python3
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from regionx import Polygon as regionx_polygon  # avoid name clash
from scipy.spatial import cKDTree

import pyarrow as pa
import pyarrow.parquet as pq


# Notes:
# - All RA are normalized to [0, 360).
# - Random Dec sampling is equal-area within the region's Dec bounds.
# - Aperture masking uses KD-tree on the randoms (unit vectors) with chord-distance radii.
# - Polygon masking is optional; by default, NGC mask is NONE (no polygon masks loaded/applied).


def normalize_ra(ra_deg: np.ndarray) -> np.ndarray:
    """Map RA to [0, 360)."""
    return np.mod(ra_deg, 360.0)


def sample_ra_wrapped(rng: np.random.Generator, ramin: float, ramax: float, n: int) -> np.ndarray:
    """
    Sample RA uniformly from an interval that may wrap around 0/360.
    Inputs can be in any convention (e.g., negative RA); output is [0, 360).
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
    """Convert RA/Dec in degrees to 3D unit vectors (x,y,z)."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z)).astype(np.float64, copy=False)


@dataclass
class ApertureMaskRow:
    ra: float
    dec: float
    radius_deg: float


@dataclass
class PolygonMaskRow:
    ra_vertices: List[float]
    dec_vertices: List[float]


def _hex_vertices_to_limits(ra_vertices: List[float], dec_vertices: List[float]) -> Dict[str, float]:
    """
    Conservative bounding box limits for sampling.
    NOTE: This is just for proposal sampling; the polygon itself is applied afterwards.
    Handles RA wrap by choosing the smallest arc span that contains the vertices.
    """
    ra = np.mod(np.asarray(ra_vertices, dtype=float), 360.0)
    dec = np.asarray(dec_vertices, dtype=float)

    # Find minimal RA span on a circle:
    # Sort and find largest gap; interval is the complement.
    ra_sorted = np.sort(ra)
    gaps = np.diff(np.concatenate([ra_sorted, ra_sorted[:1] + 360.0]))
    k = int(np.argmax(gaps))
    # interval start/end in "unwrapped" coordinates
    start = ra_sorted[(k + 1) % ra_sorted.size]
    end = ra_sorted[k] + (360.0 if k == ra_sorted.size - 1 else 0.0)

    # Convert to (ramin, ramax) possibly wrapped
    ramin = float(start % 360.0)
    ramax = float(end % 360.0)

    return {
        "ramin": ramin,
        "ramax": ramax,
        "decmin": float(dec.min()),
        "decmax": float(dec.max()),
    }


class WDHexRandoms:
    def __init__(
        self,
        seed: int = 42,
        nrandoms: int = 50_000_000,
        starmask_path: str = "./masks/starmask.dat",
        ghostmask_path: str = "./masks/ghostmask.dat",
        polygonmask_path: Optional[str] = None,  # default NONE
        save_location: str = "./wd_randoms/",
        chunk_size: int = 1_000_000,
    ):
        self.rng = np.random.default_rng(seed)
        self.nrandoms = int(nrandoms)
        self.chunk_size = int(chunk_size)

        # Hex regions from user-provided vertices
        self.region_polygons: Dict[str, PolygonMaskRow] = {
            "WD01": PolygonMaskRow(
                ra_vertices=[9.500000, 11.007814, 11.040298, 9.500000, 7.959702, 7.992186],
                dec_vertices=[-42.683240, -43.316620, -44.583380, -45.216760, -44.583380, -43.316620],
            ),
            "WD02": PolygonMaskRow(
                ra_vertices=[35.240129, 36.509871, 37.146647, 36.512013, 35.237987, 34.603353],
                dec_vertices=[-3.927954, -3.927954, -5.025000, -6.122046, -6.122046, -5.025000],
            ),
            "WD03": PolygonMaskRow(
                ra_vertices=[52.414122, 53.835877, 54.561029, 53.850565, 52.399435, 51.688971],
                dec_vertices=[-27.002954, -27.002954, -28.100000, -29.197046, -29.197046, -28.100000],
            ),
            "WD10": PolygonMaskRow(
                ra_vertices=[149.490570, 150.759430, 151.392694, 150.758497, 149.491503, 148.857306],
                dec_vertices=[3.297046, 3.297046, 2.200000, 1.102954, 1.102954, 2.200000],
            ),
        }

        # Sampling limits (bounding boxes)
        self.regions: Dict[str, Dict[str, float]] = {
            name: _hex_vertices_to_limits(poly.ra_vertices, poly.dec_vertices)
            for name, poly in self.region_polygons.items()
        }

        self.region_randoms = {
            region: {"ra": None, "dec": None, "in_region": None, "starmask": None, "ghostmask": None, "polygon_mask": None, "realisation": None}
            for region in self.regions.keys()
        }

        # Mask catalogs
        self.star_mask_catalog: List[ApertureMaskRow] = []
        self.ghostmask_catalog: List[ApertureMaskRow] = []
        self.polygon_catalog: List[PolygonMaskRow] = []  # optional extra polygon masks (e.g., NGC), default none

        self.starmask_path = starmask_path
        self.ghostmask_path = ghostmask_path
        self.polygonmask_path = polygonmask_path
        self.save_location = save_location
        os.makedirs(self.save_location, exist_ok=True)

    # -----------------------------
    # Region generation
    # -----------------------------
    def generate_randoms(self, region: str, nrandoms: Optional[int] = None) -> None:
        """
        Generate random points inside the hex region (exact polygon), using
        proposal draws from a bounding box and then polygon acceptance.

        Stores:
          - ra, dec (accepted points)
          - in_region (all True for accepted points, kept for completeness)
          - mask columns initialized False
        """
        if region not in self.regions:
            raise KeyError(f"Unknown region '{region}'. Known: {list(self.regions.keys())}")

        n_target = self.nrandoms if nrandoms is None else int(nrandoms)
        lim = self.regions[region]
        poly_row = self.region_polygons[region]
        poly = regionx_polygon([v % 360.0 for v in poly_row.ra_vertices], poly_row.dec_vertices)

        ra_out = np.empty(n_target, dtype=np.float64)
        dec_out = np.empty(n_target, dtype=np.float64)

        filled = 0
        # Heuristic oversample factor; hex in its bbox is ~0.65–0.75 typically.
        # We'll loop until filled.
        while filled < n_target:
            need = n_target - filled
            draw = int(np.ceil(need * 1.6))

            ra_try = sample_ra_wrapped(self.rng, lim["ramin"], lim["ramax"], draw)
            dec_try = sample_dec_equal_area(self.rng, lim["decmin"], lim["decmax"], draw)

            inside = np.asarray(poly.check_points(ra_try.tolist(), dec_try.tolist()), dtype=bool)
            n_acc = int(inside.sum())
            if n_acc == 0:
                continue

            take = min(need, n_acc)
            ra_out[filled : filled + take] = ra_try[inside][:take]
            dec_out[filled : filled + take] = dec_try[inside][:take]
            filled += take

        self.region_randoms[region]["ra"] = ra_out
        self.region_randoms[region]["dec"] = dec_out
        self.region_randoms[region]["in_region"] = np.ones(n_target, dtype=bool)

        # Initialise mask columns (False means "not masked")
        self.region_randoms[region]["starmask"] = np.zeros(n_target, dtype=bool)
        self.region_randoms[region]["ghostmask"] = np.zeros(n_target, dtype=bool)
        self.region_randoms[region]["polygon_mask"] = np.zeros(n_target, dtype=bool)
        self.region_randoms[region]["realisation"] = np.zeros(n_target, dtype=np.int32)

    # -----------------------------
    # Mask loading
    # -----------------------------
    def _load_aperture_mask_csv(self, path: str) -> List[ApertureMaskRow]:
        """
        Loads a whitespace-separated file with 3 columns per line:
            ra  dec  radius_deg
        No header. Lines starting with '#' are ignored.
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

                parts = s.split()
                if len(parts) < 3:
                    bad_lines += 1
                    continue

                try:
                    ra = float(parts[0]) % 360.0
                    dec = float(parts[1])
                    rad = float(parts[2])
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno} could not parse floats: {s}") from e

                rows.append(ApertureMaskRow(ra=ra, dec=dec, radius_deg=rad))

        if len(rows) == 0:
            raise RuntimeError(f"Loaded 0 apertures from {path}. Check delimiter/format. (bad_lines={bad_lines})")

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
            raise RuntimeError(f"Loaded 0 polygons from {path}. Check delimiter/format. (bad_lines={bad_lines})")

        print(f"Loaded {len(polys)} polygons from {path} (skipped {bad_lines} malformed lines)")
        return polys

    def load_masks(self) -> None:
        """Loads star + ghost masks, and optional polygon mask file (default: none)."""
        self.star_mask_catalog = self._load_aperture_mask_csv(self.starmask_path)
        self.ghostmask_catalog = self._load_aperture_mask_csv(self.ghostmask_path)

        if self.polygonmask_path is None:
            self.polygon_catalog = []
            print("Polygon mask: NONE (default)")
        else:
            self.polygon_catalog = self._load_polygon_mask_csv(self.polygonmask_path)

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

        ap_ra = np.array([row.ra for row in catalog], dtype=np.float64)
        ap_dec = np.array([row.dec for row in catalog], dtype=np.float64)
        ap_rad_deg = np.array([row.radius_deg for row in catalog], dtype=np.float64)

        ap_xyz = radec_to_unitvec(ap_ra, ap_dec)  # (M, 3)
        theta = np.deg2rad(ap_rad_deg)
        ap_r = 2.0 * np.sin(theta / 2.0)          # chord distances

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)

            xyz = radec_to_unitvec(ra[start:end], dec[start:end])  # (Nc, 3)
            tree = cKDTree(xyz, leafsize=64)

            chunk_mask = np.zeros(end - start, dtype=bool)

            for i in range(ap_xyz.shape[0]):
                if chunk_mask.all():
                    break
                idx = tree.query_ball_point(ap_xyz[i], r=ap_r[i])
                if idx:
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
        Applies optional polygon masks (e.g., NGC). Default is NONE, so does nothing unless a file is provided.
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

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)

            ra_list = ra[start:end].tolist()
            dec_list = dec[start:end].tolist()

            chunk_mask = np.zeros(end - start, dtype=bool)
            for prow in self.polygon_catalog:
                poly = regionx_polygon([v % 360.0 for v in prow.ra_vertices], prow.dec_vertices)
                chunk_mask |= np.asarray(poly.check_points(ra_list, dec_list), dtype=bool)
                if chunk_mask.all():
                    break

            out[start:end] = chunk_mask

        self.region_randoms[region]["polygon_mask"] = out

    # -----------------------------
    # Extra columns + save
    # -----------------------------
    def add_realisation_flag(self, region: str, block_size: int = 10_000_000) -> None:
        """Realisation flag: starts at 0 and increases by 1 every block_size entries."""
        ra = self.region_randoms[region]["ra"]
        if ra is None:
            raise RuntimeError(f"No randoms generated for region '{region}'")

        n = ra.size
        self.region_randoms[region]["realisation"] = (np.arange(n, dtype=np.int64) // int(block_size)).astype(np.int32)

    def save_randoms_parquet_chunked(self, region: str) -> str:
        data = self.region_randoms[region]
        ra = data["ra"]
        dec = data["dec"]
        if ra is None or dec is None:
            raise RuntimeError(f"No randoms generated for region '{region}'")

        outpath = os.path.join(self.save_location, f"{region}_randoms.parquet")

        schema = pa.schema(
            [
                ("ra", pa.float64()),
                ("dec", pa.float64()),
                ("in_region", pa.bool_()),
                ("starmask", pa.bool_()),
                ("ghostmask", pa.bool_()),
                ("polygon_mask", pa.bool_()),
                ("realisation", pa.int32()),
            ]
        )

        writer = pq.ParquetWriter(outpath, schema=schema, compression="zstd")
        try:
            n = ra.size
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)

                batch = pa.table(
                    {
                        "ra": ra[start:end].astype(np.float64),
                        "dec": dec[start:end].astype(np.float64),
                        "in_region": data["in_region"][start:end].astype(bool),
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
    parser = argparse.ArgumentParser(description="Generate WD hex-region randoms and apply aperture masks (optional polygons).")
    parser.add_argument("--region", default="WD01", choices=["WD01", "WD02", "WD03", "WD10"])
    parser.add_argument("--nrandoms", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--save-location", type=str, default="./wd_randoms/")

    # Masks
    parser.add_argument("--starmask-path", type=str, default="./masks/starmask.dat")
    parser.add_argument("--ghostmask-path", type=str, default="./masks/ghostmask.dat")
    parser.add_argument(
        "--polygonmask-path",
        type=str,
        default=None,  # IMPORTANT: default none
        help="Optional polygon mask file (whitespace polygons). Default: none.",
    )

    parser.add_argument("--no-masks", action="store_true", help="Skip loading/applying masks entirely")
    parser.add_argument("--no-star", action="store_true", help="Skip applying star mask")
    parser.add_argument("--no-ghost", action="store_true", help="Skip applying ghost mask")
    parser.add_argument("--no-polygon", action="store_true", help="Skip applying polygon mask")

    parser.add_argument("--block-size", type=int, default=10_000_000, help="Realisation block size (default 10M)")

    args = parser.parse_args()

    gen = WDHexRandoms(
        seed=args.seed,
        nrandoms=args.nrandoms,
        starmask_path=args.starmask_path,
        ghostmask_path=args.ghostmask_path,
        polygonmask_path=args.polygonmask_path,  # default None => none
        save_location=args.save_location,
        chunk_size=args.chunk_size,
    )

    # Decide which regions to run
    regions = [args.region]

    if not args.no_masks:
        gen.load_masks()

    for region in regions:
        print(f"[{region}] generating {args.nrandoms} randoms (hex polygon accept/reject)")
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

        sm = gen.region_randoms[region]["starmask"]
        gm = gen.region_randoms[region]["ghostmask"]
        pm = gen.region_randoms[region]["polygon_mask"]
        print(f"[{region}] masked fractions: star={sm.mean():.4f}, ghost={gm.mean():.4f}, poly={pm.mean():.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
