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
    def __init__(self, filename, region= None, realisation_column='realisation'):
        self.filename = filename
        self.realisation_column = realisation_column
        self.columns_to_count = ["starmask", "ghostmask", "polygonmask"]

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
        

    def find_area_rectange_on_sky(self):
        "Find the area of a rectangle on the sky given by ra/dec limits in sq deg"
        return None
    
    def find_area_hexagon_on_sky(self):
        "Find the area of a hexagon on the sky given by ra/dec vertices in sq deg"
        return None

    def find_area_fractions_realisationwise(self):
        # Read in realisation column from parquet file
        table = pq.read_table(self.filename, columns=[self.realisation_column])
        # Find unique realisations
        realisations = table[self.realisation_column].to_numpy()
        unique_realisations = np.unique(realisations)
        del table
        del realisations

        for realisation in unique_realisations:
            print(f"Processing realisation: {realisation}")
            # Read in relevant columns for this realisation
            filter_expression = (pa.compute.equal(pa.array(realisations), realisation))
            table = pq.read_table(self.filename, columns=self.columns_to_count, filters=[(self.realisation_column, "=", realisation)])

            # Count the number of pixels for each mask type
            # Columns to count should all be boolean
            iteration_count = table.num_rows

            self.total_count += iteration_count
        print(f"Total count across all realisations: {self.total_count}")

    def find_areas_in_sq_deg(self):
        """Find the areas of the regions in sq deg, split by:
        - total area
        - area masked in starmask
        - area masked in starmask or ghostmask
        - area masked in starmask or polygonmask
        - area masked in starmask or ghostmask or polygonmask
        This is done by getting the total area in sq deg and multiplying by the fractions found in find_area_fractions_realisationwise.
        Also print out errors, assuming possible binomial errors on the fractions.

        """
        return None


    



def main():
    parser = argparse.ArgumentParser(description="Find areas of sky regions from polygon masks.")
    parser.add_argument("filename", type=str, help="Path to the Parquet file containing polygon masks.")
    parser.add_argument("--region", type=str, default=None, help="Region to calculate area for (e.g., WD01, WD02, WD03, WD10). If None, calculates for all regions.")
    parser.add_argument("--realisation_column", type=str, default="realisation", help="Column name for realisations in the Parquet file.")
    args = parser.parse_args()

    area_finder = AreaFinder(args.filename, region=args.region, realisation_column=args.realisation_column)
    area_finder.find_area_fractions_realisationwise()


if __name__ == "__main__":
    main()

