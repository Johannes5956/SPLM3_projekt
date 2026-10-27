# import necessary libraries

import rasterio
import matplotlib.pyplot as plt

# Define and open the path to the raster file

path = "data/traning/overflade_1.tif"

with rasterio.open(path) as src:
    print("Bredde:", src.width)
    print("Højde:", src.height)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Opløsning:", src.res)
    print("NoData:", src.nodata)

    elevation = src.read(1)

print("Min højde:", elevation.min())
print("Maks højde:", elevation.max())







