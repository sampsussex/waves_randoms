import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

@dataclass
class PolygonMaskRow:
    ra_vertices: List[float]
    dec_vertices: List[float]


class AreaFinder:
    def __init__(self, filename, region=None, realisation_column="realisation"):
        self.filename = filename
        self.region = region
        self.realisation_column = realisation_column
        self.columns_to_count = ["starmask", "ghostmask", "polygon_mask"]

        self.region_ddfs: Dict[str, PolygonMaskRow] = {
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


        waves_n_limits = {"ramax": 225.0, "decmax": 3.95, "ramin": 157.25, "decmin": -3.95}
        waves_s_limits = {"ramax": 51.6, "decmax": -27.0, "ramin": -30.0, "decmin": -35.6}
        waves_deep_limits = {"ramax": 351, "decmax": -30 ,"ramin": 339, "decmin":-35}

        # region can either be a ddf (hexagon) or wide/deep region (rectangle)

        self.regions_wide: Dict[str, Dict[str, float]] = {
            "waves-wide_n": waves_n_limits,
            "waves-wide_s": waves_s_limits,
            "waves-deep": waves_deep_limits,
        }

        # Need for:
        # masked in starmask
        # masked in starmask or ghostmask
        # masked in starmask or polygon
        # masked in starmask pr polygon or ghostmask

        self.total_count = None
        self.starmask_count = None
        self.starmask_ghostmask_count = None
        self.starmask_polygonmask_count = None
        self.starmask_ghostmask_polygonmask_count = None

    @staticmethod
    def _wrap_longitude_diff(delta_lon: float) -> float:
        while delta_lon <= -np.pi:
            delta_lon += 2.0 * np.pi
        while delta_lon > np.pi:
            delta_lon -= 2.0 * np.pi
        return delta_lon

    def find_area_rectange_on_sky(self, ramin: float, ramax: float, decmin: float, decmax: float) -> float:
        "Find the area of a rectangle on the sky given by ra/dec limits in sq deg"
        if ramax < ramin:
            ramax += 360.0
        ra1 = np.deg2rad(ramin)
        ra2 = np.deg2rad(ramax)
        dec1 = np.deg2rad(decmin)
        dec2 = np.deg2rad(decmax)
        area_sr = (ra2 - ra1) * (np.sin(dec2) - np.sin(dec1))
        return area_sr * (180.0 / np.pi) ** 2

    def find_area_hexagon_on_sky(self, ra_vertices: List[float], dec_vertices: List[float]) -> float:
        "Find the area of a hexagon on the sky given by ra/dec vertices in sq deg"
        if len(ra_vertices) != len(dec_vertices):
            raise ValueError("RA/Dec vertex lists must be the same length.")
        if len(ra_vertices) < 3:
            raise ValueError("Need at least 3 vertices to define a polygon.")

        lons = np.deg2rad(np.array(ra_vertices, dtype=float))
        lats = np.deg2rad(np.array(dec_vertices, dtype=float))
        if lons[0] != lons[-1] or lats[0] != lats[-1]:
            lons = np.append(lons, lons[0])
            lats = np.append(lats, lats[0])

        total = 0.0
        for idx in range(len(lons) - 1):
            lon1 = lons[idx]
            lon2 = lons[idx + 1]
            lat1 = lats[idx]
            lat2 = lats[idx + 1]
            delta_lon = self._wrap_longitude_diff(lon2 - lon1)
            total += delta_lon * (2.0 + np.sin(lat1) + np.sin(lat2))

        area_sr = abs(total) / 2.0
        return area_sr * (180.0 / np.pi) ** 2

    def find_area_fractions_realisationwise(self):
        # Read in realisation column from parquet file
        parquet_file = pq.ParquetFile(self.filename)
        available_columns = set(parquet_file.schema.names)
        columns_to_count = [col for col in self.columns_to_count if col in available_columns]
        if len(columns_to_count) != len(self.columns_to_count):
            missing = sorted(set(self.columns_to_count) - set(columns_to_count))
            print(f"Warning: missing columns {missing}. Treating as all-false.")

        table = pq.read_table(self.filename, columns=[self.realisation_column])
        # Find unique realisations
        realisations = table[self.realisation_column].to_numpy()
        unique_realisations = np.unique(realisations)
        del table
        del realisations

        self.total_count = 0
        self.starmask_count = 0
        self.starmask_ghostmask_count = 0
        self.starmask_polygonmask_count = 0
        self.starmask_ghostmask_polygonmask_count = 0

        for realisation in unique_realisations:
            print(f"Processing realisation: {realisation}")
            # Read in relevant columns for this realisation
            table = pq.read_table(
                self.filename,
                columns=columns_to_count,
                filters=[(self.realisation_column, "=", realisation)],
            )

            # Count the number of pixels for each mask type
            # Columns to count should all be boolean
            iteration_count = table.num_rows

            starmask = table["starmask"].to_numpy(zero_copy_only=False).astype(bool)
            ghostmask = table["ghostmask"].to_numpy(zero_copy_only=False).astype(bool)
            if "polygon_mask" in table.column_names:
                polygonmask = table["polygon_mask"].to_numpy(zero_copy_only=False).astype(bool)
            else:
                polygonmask = np.zeros(iteration_count, dtype=bool)

            self.total_count += iteration_count
            self.starmask_count += int(np.count_nonzero(starmask))
            self.starmask_ghostmask_count += int(np.count_nonzero(starmask | ghostmask))
            self.starmask_polygonmask_count += int(np.count_nonzero(starmask | polygonmask))
            self.starmask_ghostmask_polygonmask_count += int(
                np.count_nonzero(starmask | ghostmask | polygonmask)
            )
        print(f"Total count across all realisations: {self.total_count}")

    def find_areas_in_sq_deg(self):
        """Find the areas of the regions in sq deg, split by:
        - total area
        - area masked in starmask
        - area masked in starmask or ghostmask
        - area masked in starmask or polygonmask
        - area masked in starmask or ghostmask or polygonmask
        This is done by getting the total area in sq deg and multiplying by the fractions found in find_area_fractions_realisationwise.
        Also print out errors, assuming a Poisson approximation for masked counts.

        """
        if self.total_count is None or self.total_count == 0:
            raise RuntimeError("Counts not initialized. Run find_area_fractions_realisationwise() first.")

        fractions = {
            "starmask": self.starmask_count / self.total_count,
            "starmask+ghostmask": self.starmask_ghostmask_count / self.total_count,
            "starmask+polygonmask": self.starmask_polygonmask_count / self.total_count,
            "starmask+ghostmask+polygonmask": self.starmask_ghostmask_polygonmask_count / self.total_count,
        }

        masked_counts = {
            "starmask": self.starmask_count,
            "starmask+ghostmask": self.starmask_ghostmask_count,
            "starmask+polygonmask": self.starmask_polygonmask_count,
            "starmask+ghostmask+polygonmask": self.starmask_ghostmask_polygonmask_count,
        }

        errors = {
            key: (self.total_count / (count ** 1.5)) if count > 0 else np.nan
            for key, count in masked_counts.items()
        }

        regions = {}
        if self.region is not None:
            if self.region in self.region_ddfs:
                regions[self.region] = ("ddf", self.region_ddfs[self.region])
            elif self.region in self.regions_wide:
                regions[self.region] = ("rect", self.regions_wide[self.region])
            else:
                raise ValueError(f"Unknown region {self.region}.")
        else:
            regions.update({name: ("ddf", row) for name, row in self.region_ddfs.items()})
            regions.update({name: ("rect", row) for name, row in self.regions_wide.items()})

        for name, (region_type, region_data) in regions.items():
            if region_type == "ddf":
                total_area = self.find_area_hexagon_on_sky(
                    region_data.ra_vertices, region_data.dec_vertices
                )
            else:
                total_area = self.find_area_rectange_on_sky(
                    region_data["ramin"],
                    region_data["ramax"],
                    region_data["decmin"],
                    region_data["decmax"],
                )

            print(f"Region {name}: total area = {total_area:.6f} sq deg")
            for key, fraction in fractions.items():
                area = total_area * fraction
                area_error = total_area * errors[key]
                print(
                    f"  {key}: {area:.6f} sq deg (fraction={fraction:.6e} ± {errors[key]:.6e}, "
                    f"area_err={area_error:.6f})"
                )






def main():
    parser = argparse.ArgumentParser(description="Find areas of sky regions from polygon masks.")
    parser.add_argument("filename", type=str, help="Path to the Parquet file containing polygon masks.")
    parser.add_argument("--region", type=str, default=None, help="Region to calculate area for (e.g., WD01, WD02, WD03, WD10). If None, calculates for all regions.")
    parser.add_argument("--realisation_column", type=str, default="realisation", help="Column name for realisations in the Parquet file.")
    args = parser.parse_args()

    area_finder = AreaFinder(args.filename, region=args.region, realisation_column=args.realisation_column)
    area_finder.find_area_fractions_realisationwise()
    area_finder.find_areas_in_sq_deg()


if __name__ == "__main__":
    main()
