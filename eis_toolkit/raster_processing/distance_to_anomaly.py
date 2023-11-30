# This is in draft stage, not yet completed
# Author: Tomi Rönkkö / GTK

# Import rasterio and numpy libraries
import rasterio
import numpy as np

# Define treshold dummy value
threshold = 5

# Open the input raster file
with rasterio.open("../../tests/data/remote/small_raster.tif") as src:
    # Read the raster data as numpy array
    data = src.read(1)
    
    # Get the raster profile (metadata)
    profile = src.profile
    
    # Create a mask for values above the threshold
    mask = data > threshold
    
    # Apply the mask to the data and set the masked values to nodata
    data[mask] = profile["nodata"]
    
    # Write the masked data to a new raster file
    with rasterio.open("masked.tif", "w", **profile) as dst:
        dst.write(data, 1)
        
# Open the masked raster file
with rasterio.open("masked.tif") as src:
    # Read the masked data as a numpy array
    data = src.read(1)
    
    # Get the raster profile (metadata)
    profile = src.profile
    
    # Get the nodata value
    nodata = profile["nodata"]
    
    # Create an empty array for the cost values
    cost = np.zeros_like(data)
    
    # Loop through each row and column of the data array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            
            # If the data value is nodata, set the cost value to zero
            if data[i, j] != nodata:
                cost[i, j] = 0
            
            # Otherwise, find the nearest value above the in the data array
            else:
                
                # Initialize the minimum distance and value variables
                min_dist = np.inf
                min_val = np.inf
                
                # Loop through each row and column of the data array again
                for k in range(data.shape[0]):
                    for l in range(data.shape[1]):
                        
                        # If the data value is above the threshold, calculate the distance to the current cell
                        if data[k, l] > threshold:
                            dist = np.sqrt((i - k) ** 2 + (j - l) ** 2)
                            
                            # If the distance is smaller than the minimum distance, update the minimum distance and value variables
                            if dist < min_dist:
                                min_dist = dist
                                min_val = data[k, l]
    
                # Set the cost value to the difference between the minimum value and the threshold value
                cost[i, j] = min_val - threshold
    
    # Write the cost data to a new raster file
    with rasterio.open("cost.tif", "w", **profile) as dst:
        dst.write(cost, 1)
    
    
    