# import necessary libraries

import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
# Define and open the path to the raster file

gdf = gpd.read_file("data/training/omrids_1.gpkg")

print(gdf.total_bounds)


from pathlib import Path
import geopandas as gpd

gdf = gpd.read_file("data/training/omrids_1.gpkg")
minx, miny, maxx, maxy = gdf.total_bounds

for laz in Path("data/training/lidar").glob("*.laz"):
    parts = laz.stem.split("_")

    tile_y = int(parts[-2])
    tile_x = int(parts[-1])

    tile_minx = tile_x * 1000
    tile_maxx = tile_minx + 1000

    tile_miny = tile_y * 1000
    tile_maxy = tile_miny + 1000

    overlap = (
        tile_minx <= maxx and
        tile_maxx >= minx and
        tile_miny <= maxy and
        tile_maxy >= miny
    )

    if overlap:
        print(laz.name)
