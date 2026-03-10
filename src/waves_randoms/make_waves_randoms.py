#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from regionx import Polygon as regionx_polygon
from scipy.spatial import cKDTree


def normalize_ra(ra_deg: np.ndarray) -> np.ndarray:
    return np.mod(ra_deg, 360.0)


def sample_ra_wrapped(rng: np.random.Generator, ramin: float, ramax: float, n: int) -> np.ndarray:
    ramin_n = float(ramin) % 360.0
    ramax_n = float(ramax) % 360.0

    if np.isclose(ramin_n, ramax_n):
        raise ValueError(f"RA range has ~zero width after normalization: ramin={ramin}, ramax={ramax}")

    if ramin_n < ramax_n:
        return rng.uniform(ramin_n, ramax_n, size=n)

    width = (360.0 - ramin_n) + ramax_n
    return normalize_ra(ramin_n + rng.random(size=n) * width)


def sample_dec_equal_area(rng: np.random.Generator, decmin: float, decmax: float, n: int) -> np.ndarray:
    if decmin >= decmax:
        raise ValueError(f"Invalid Dec limits: decmin={decmin} >= decmax={decmax}")
    if decmin < -90.0 or decmax > 90.0:
        raise ValueError(f"Dec limits out of bounds [-90, 90]: decmin={decmin}, decmax={decmax}")

    smin = np.sin(np.deg2rad(decmin))
    smax = np.sin(np.deg2rad(decmax))
    return np.rad2deg(np.arcsin(rng.uniform(smin, smax, size=n)))


def in_ra_range_wrapped(ra_deg: np.ndarray, ramin: float, ramax: float) -> np.ndarray:
    ra_n = normalize_ra(np.asarray(ra_deg, dtype=np.float64))
    ramin_n = float(ramin) % 360.0
    ramax_n = float(ramax) % 360.0
    if ramin_n < ramax_n:
        return (ra_n >= ramin_n) & (ra_n <= ramax_n)
    return (ra_n >= ramin_n) | (ra_n <= ramax_n)


def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    return np.column_stack((cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec))).astype(np.float64, copy=False)


def polygon_ball_radius_chord(ra_vertices: List[float], dec_vertices: List[float]) -> Tuple[np.ndarray, float]:
    vertices_xyz = radec_to_unitvec(np.asarray(ra_vertices, dtype=np.float64), np.asarray(dec_vertices, dtype=np.float64))
    first_xyz = vertices_xyz[0]
    return first_xyz, float(np.linalg.norm(vertices_xyz - first_xyz, axis=1).max())


def hex_vertices_to_limits(ra_vertices: List[float], dec_vertices: List[float]) -> Dict[str, float]:
    ra = np.mod(np.asarray(ra_vertices, dtype=float), 360.0)
    dec = np.asarray(dec_vertices, dtype=float)

    ra_sorted = np.sort(ra)
    gaps = np.diff(np.concatenate([ra_sorted, ra_sorted[:1] + 360.0]))
    k = int(np.argmax(gaps))
    start = ra_sorted[(k + 1) % ra_sorted.size]
    end = ra_sorted[k] + (360.0 if k == ra_sorted.size - 1 else 0.0)

    return {
        "ramin": float(start % 360.0),
        "ramax": float(end % 360.0),
        "decmin": float(dec.min()),
        "decmax": float(dec.max()),
    }


@dataclass
class ApertureMaskRow:
    ra: float
    dec: float
    radius_deg: float


@dataclass
class PolygonMaskRow:
    ra_vertices: List[float]
    dec_vertices: List[float]


@dataclass
class RegionConfig:
    mode: str
    limits: Dict[str, float]
    polygon_vertices: Optional[PolygonMaskRow]
    default_star: Optional[str]
    default_ghost: Optional[str]
    default_polygon: Optional[str]
    default_extra: Optional[str]
    ghost_radius_scale: float = 1.0


class WavesRandomGenerator:
    def __init__(self, seed: int, nrandoms: int, chunk_size: int, save_location: str):
        self.rng = np.random.default_rng(seed)
        self.nrandoms = int(nrandoms)
        self.chunk_size = int(chunk_size)
        self.save_location = save_location
        os.makedirs(self.save_location, exist_ok=True)

        wd_polygons = {
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

        self.regions: Dict[str, RegionConfig] = {
            "waves-wide_n": RegionConfig(
                mode="box",
                limits={"ramax": 225.0, "decmax": 3.95, "ramin": 157.25, "decmin": -3.95},
                polygon_vertices=None,
                default_star="./masks/starmask_waves_n.dat",
                default_ghost="./masks/ghostmask_waves_n.dat",
                default_polygon="./masks/ngc_n.dat",
                default_extra=None,
            ),
            "waves-wide_s": RegionConfig(
                mode="box",
                limits={"ramax": 51.6, "decmax": -27.0, "ramin": -30.0, "decmin": -35.6},
                polygon_vertices=None,
                default_star="./masks/starmask_waves_s.dat",
                default_ghost="./masks/ghostmask_waves_s.dat",
                default_polygon="./masks/ngc_s.dat",
                default_extra="./masks/extra_waves_s_sources.dat",
            ),
            "waves-deep": RegionConfig(
                mode="box",
                limits={"ramax": 351.0, "decmax": -30.0, "ramin": 339.0, "decmin": -35.0},
                polygon_vertices=None,
                default_star="./masks/starmask_waves_s.dat",
                default_ghost="./masks/ghostmask_waves_s.dat",
                default_polygon="./masks/ngc_s.dat",
                default_extra="./masks/extra_waves_s_sources.dat",
            ),
            "WD01": RegionConfig("hex", hex_vertices_to_limits(wd_polygons["WD01"].ra_vertices, wd_polygons["WD01"].dec_vertices), wd_polygons["WD01"], "./masks/WD01_stars.dat", "./masks/WD01_ghosts.dat", None, None, 1.0 / 60.0),
            "WD02": RegionConfig("hex", hex_vertices_to_limits(wd_polygons["WD02"].ra_vertices, wd_polygons["WD02"].dec_vertices), wd_polygons["WD02"], "./masks/WD02_stars.dat", "./masks/WD02_ghosts.dat", None, None, 1.0 / 60.0),
            "WD03": RegionConfig("hex", hex_vertices_to_limits(wd_polygons["WD03"].ra_vertices, wd_polygons["WD03"].dec_vertices), wd_polygons["WD03"], "./masks/WD03_stars.dat", "./masks/WD03_ghosts.dat", None, None, 1.0 / 60.0),
            "WD10": RegionConfig("hex", hex_vertices_to_limits(wd_polygons["WD10"].ra_vertices, wd_polygons["WD10"].dec_vertices), wd_polygons["WD10"], "./masks/WD10_stars.dat", "./masks/WD10_ghosts.dat", None, None, 1.0 / 60.0),
        }

        self.region_randoms = {
            region: {"ra": None, "dec": None, "in_region": None, "starmask": None, "ghostmask": None, "polygon_mask": None, "realisation": None}
            for region in self.regions
        }

    def set_points_from_catalog(self, region: str, ra: np.ndarray, dec: np.ndarray) -> None:
        cfg = self.regions[region]
        ra = normalize_ra(np.asarray(ra, dtype=np.float64))
        dec = np.asarray(dec, dtype=np.float64)
        if ra.shape != dec.shape:
            raise ValueError(f"RA/Dec arrays must be the same shape: {ra.shape} vs {dec.shape}")

        if cfg.mode == "box":
            lim = cfg.limits
            in_region = in_ra_range_wrapped(ra, lim["ramin"], lim["ramax"]) & (dec >= lim["decmin"]) & (dec <= lim["decmax"])
        else:
            poly = regionx_polygon([v % 360.0 for v in cfg.polygon_vertices.ra_vertices], cfg.polygon_vertices.dec_vertices)
            in_region = np.asarray(poly.check_points(ra.tolist(), dec.tolist()), dtype=bool)

        n = ra.size
        self.region_randoms[region].update(
            {
                "ra": ra,
                "dec": dec,
                "in_region": in_region,
                "starmask": np.zeros(n, dtype=bool),
                "ghostmask": np.zeros(n, dtype=bool),
                "polygon_mask": np.zeros(n, dtype=bool),
                "realisation": np.zeros(n, dtype=np.int32),
            }
        )

    def load_parquet_radec(self, path: str, ra_column: str, dec_column: str) -> Tuple[np.ndarray, np.ndarray]:
        table = pq.read_table(path, columns=[ra_column, dec_column])
        if ra_column not in table.column_names or dec_column not in table.column_names:
            raise KeyError(f"Parquet file missing requested columns: {ra_column}, {dec_column}")
        ra = table[ra_column].to_numpy(zero_copy_only=False)
        dec = table[dec_column].to_numpy(zero_copy_only=False)
        return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)

    def _load_aperture_mask_csv(self, path: str, radius_scale: float = 1.0) -> List[ApertureMaskRow]:
        rows: List[ApertureMaskRow] = []
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "r") as f:
            for lineno, line in enumerate(f, start=1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    ra = float(parts[0]) % 360.0
                    dec = float(parts[1])
                    rad = float(parts[2]) * radius_scale
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno} parse error: {s}") from e
                rows.append(ApertureMaskRow(ra=ra, dec=dec, radius_deg=rad))

        if not rows:
            raise RuntimeError(f"Loaded 0 apertures from {path}")
        return rows

    def _load_polygon_mask_csv(self, path: str) -> List[PolygonMaskRow]:
        polys: List[PolygonMaskRow] = []
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            for lineno, line in enumerate(f, start=1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 6 or len(parts) % 2 != 0:
                    continue
                try:
                    vals = [float(x) for x in parts]
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno} parse error: {s}") from e
                polys.append(PolygonMaskRow(
                    ra_vertices=[vals[i] % 360.0 for i in range(0, len(vals), 2)],
                    dec_vertices=[vals[i] for i in range(1, len(vals), 2)],
                ))
        if not polys:
            raise RuntimeError(f"Loaded 0 polygons from {path}")
        return polys

    def _resolve_mask_paths(self, region: str, args: argparse.Namespace) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], float]:
        cfg = self.regions[region]
        return (
            args.starmask_path or cfg.default_star,
            args.ghostmask_path or cfg.default_ghost,
            args.polygonmask_path if args.polygonmask_path is not None else cfg.default_polygon,
            args.extra_waves_s_masks if args.extra_waves_s_masks is not None else cfg.default_extra,
            cfg.ghost_radius_scale,
        )

    def generate_randoms(self, region: str, nrandoms: Optional[int] = None) -> None:
        cfg = self.regions[region]
        n = self.nrandoms if nrandoms is None else int(nrandoms)
        lim = cfg.limits

        if cfg.mode == "box":
            ra = sample_ra_wrapped(self.rng, lim["ramin"], lim["ramax"], n)
            dec = sample_dec_equal_area(self.rng, lim["decmin"], lim["decmax"], n)
            in_region = np.ones(n, dtype=bool)
        else:
            poly = regionx_polygon([v % 360.0 for v in cfg.polygon_vertices.ra_vertices], cfg.polygon_vertices.dec_vertices)
            ra = np.empty(n, dtype=np.float64)
            dec = np.empty(n, dtype=np.float64)
            filled = 0
            while filled < n:
                need = n - filled
                draw = int(np.ceil(need * 1.6))
                ra_try = sample_ra_wrapped(self.rng, lim["ramin"], lim["ramax"], draw)
                dec_try = sample_dec_equal_area(self.rng, lim["decmin"], lim["decmax"], draw)
                inside = np.asarray(poly.check_points(ra_try.tolist(), dec_try.tolist()), dtype=bool)
                if not inside.any():
                    continue
                take = min(need, int(inside.sum()))
                ra[filled:filled + take] = ra_try[inside][:take]
                dec[filled:filled + take] = dec_try[inside][:take]
                filled += take
            in_region = np.ones(n, dtype=bool)

        self.region_randoms[region].update(
            {
                "ra": ra,
                "dec": dec,
                "in_region": in_region,
                "starmask": np.zeros(n, dtype=bool),
                "ghostmask": np.zeros(n, dtype=bool),
                "polygon_mask": np.zeros(n, dtype=bool),
                "realisation": np.zeros(n, dtype=np.int32),
            }
        )

    def _apply_aperture_catalog_kdtree(self, ra: np.ndarray, dec: np.ndarray, catalog: List[ApertureMaskRow]) -> np.ndarray:
        out = np.zeros(ra.size, dtype=bool)
        if not catalog:
            return out

        ap_xyz = radec_to_unitvec(
            np.asarray([row.ra for row in catalog], dtype=np.float64),
            np.asarray([row.dec for row in catalog], dtype=np.float64),
        )
        ap_r = 2.0 * np.sin(np.deg2rad(np.asarray([row.radius_deg for row in catalog], dtype=np.float64)) / 2.0)

        for start in range(0, ra.size, self.chunk_size):
            end = min(start + self.chunk_size, ra.size)
            xyz = radec_to_unitvec(ra[start:end], dec[start:end])
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

    def apply_masks(self, region: str, star_catalog: List[ApertureMaskRow], ghost_catalog: List[ApertureMaskRow], polygon_catalog: List[PolygonMaskRow], extra_catalog: List[ApertureMaskRow], do_star: bool, do_ghost: bool, do_polygon: bool) -> None:
        ra = self.region_randoms[region]["ra"]
        dec = self.region_randoms[region]["dec"]
        if do_star:
            self.region_randoms[region]["starmask"] = self._apply_aperture_catalog_kdtree(ra, dec, star_catalog)
        if do_ghost:
            self.region_randoms[region]["ghostmask"] = self._apply_aperture_catalog_kdtree(ra, dec, ghost_catalog)
        if do_polygon:
            out = np.zeros(ra.size, dtype=bool)
            if polygon_catalog:
                xyz = radec_to_unitvec(ra, dec)
                tree = cKDTree(xyz, leafsize=64)
                for prow in polygon_catalog:
                    poly = regionx_polygon([v % 360.0 for v in prow.ra_vertices], prow.dec_vertices)
                    first_xyz, radius = polygon_ball_radius_chord(prow.ra_vertices, prow.dec_vertices)
                    idx = tree.query_ball_point(first_xyz, r=radius)
                    if not idx:
                        continue
                    idx_arr = np.asarray(idx, dtype=np.int64)
                    inside = np.asarray(poly.check_points(ra[idx_arr].tolist(), dec[idx_arr].tolist()), dtype=bool)
                    if inside.any():
                        out[idx_arr[inside]] = True
            if extra_catalog:
                out |= self._apply_aperture_catalog_kdtree(ra, dec, extra_catalog)
            self.region_randoms[region]["polygon_mask"] = out

    def add_realisation_flag(self, region: str, block_size: int) -> None:
        n = self.region_randoms[region]["ra"].size
        self.region_randoms[region]["realisation"] = (np.arange(n, dtype=np.int64) // int(block_size)).astype(np.int32)

    def save_randoms_parquet_chunked(self, region: str) -> str:
        data = self.region_randoms[region]
        outpath = os.path.join(self.save_location, f"{region}_randoms.parquet")
        schema = pa.schema([
            ("ra", pa.float64()),
            ("dec", pa.float64()),
            ("in_region", pa.bool_()),
            ("starmask", pa.bool_()),
            ("ghostmask", pa.bool_()),
            ("polygon_mask", pa.bool_()),
            ("realisation", pa.int32()),
        ])

        writer = pq.ParquetWriter(outpath, schema=schema, compression="zstd")
        try:
            n = data["ra"].size
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                writer.write_table(pa.table({
                    "ra": data["ra"][start:end].astype(np.float64),
                    "dec": data["dec"][start:end].astype(np.float64),
                    "in_region": data["in_region"][start:end].astype(bool),
                    "starmask": data["starmask"][start:end].astype(bool),
                    "ghostmask": data["ghostmask"][start:end].astype(bool),
                    "polygon_mask": data["polygon_mask"][start:end].astype(bool),
                    "realisation": data["realisation"][start:end].astype(np.int32),
                }, schema=schema))
        finally:
            writer.close()
        return outpath


def main() -> None:
    all_regions = ["waves-wide_n", "waves-wide_s", "waves-deep", "WD01", "WD02", "WD03", "WD10"]

    parser = argparse.ArgumentParser(description="Generate WAVES randoms for both wide/deep and WD regions in one script.")
    parser.add_argument("--region", default="waves-wide_n", choices=all_regions)
    parser.add_argument("--nrandoms", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--save-location", type=str, default="./waves_randoms/")

    parser.add_argument("--starmask-path", type=str, default=None)
    parser.add_argument("--ghostmask-path", type=str, default=None)
    parser.add_argument("--polygonmask-path", type=str, default=None)
    parser.add_argument("--extra-waves-s-masks", type=str, default=None)

    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--no-star", action="store_true")
    parser.add_argument("--no-ghost", action="store_true")
    parser.add_argument("--no-polygon", action="store_true")
    parser.add_argument("--block-size", type=int, default=10_000_000)
    parser.add_argument("--input-catalog-parquet", type=str, default=None)
    parser.add_argument("--input-ra-column", type=str, default="RAmax")
    parser.add_argument("--input-dec-column", type=str, default="Decmax")

    args = parser.parse_args()

    gen = WavesRandomGenerator(args.seed, args.nrandoms, args.chunk_size, args.save_location)
    region = args.region
    star_path, ghost_path, polygon_path, extra_path, ghost_scale = gen._resolve_mask_paths(region, args)

    star_catalog: List[ApertureMaskRow] = []
    ghost_catalog: List[ApertureMaskRow] = []
    polygon_catalog: List[PolygonMaskRow] = []
    extra_catalog: List[ApertureMaskRow] = []

    if not args.no_masks:
        if star_path:
            star_catalog = gen._load_aperture_mask_csv(star_path)
        if ghost_path:
            ghost_catalog = gen._load_aperture_mask_csv(ghost_path, radius_scale=ghost_scale)
        if polygon_path:
            polygon_catalog = gen._load_polygon_mask_csv(polygon_path)
        if extra_path:
            extra_catalog = gen._load_aperture_mask_csv(extra_path)

    if args.input_catalog_parquet:
        print(f"[{region}] loading input parquet -> {args.input_catalog_parquet}")
        input_ra, input_dec = gen.load_parquet_radec(args.input_catalog_parquet, args.input_ra_column, args.input_dec_column)
        print(f"[{region}] applying masks to {input_ra.size} input rows")
        gen.set_points_from_catalog(region, input_ra, input_dec)
    else:
        print(f"[{region}] generating {args.nrandoms} randoms")
        gen.generate_randoms(region)
    gen.add_realisation_flag(region, args.block_size)

    if not args.no_masks:
        gen.apply_masks(
            region,
            star_catalog,
            ghost_catalog,
            polygon_catalog,
            extra_catalog,
            do_star=not args.no_star,
            do_ghost=not args.no_ghost,
            do_polygon=not args.no_polygon,
        )

    outpath = gen.save_randoms_parquet_chunked(region)
    print(f"[{region}] saved -> {outpath}")


if __name__ == "__main__":
    main()
