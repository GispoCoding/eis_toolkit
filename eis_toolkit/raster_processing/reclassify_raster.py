from typing import List, Optional

import numpy as np
import mapclassify as mc
from jenkspy import JenksNaturalBreaks
import math
import rasterio

def raster_with_manual_breaks(
    raster: rasterio.io.DatasetReader,
    path_to_file: str,
    breaks: List[int],
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
   
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    with rasterio.open(path_to_file, 'w', **raster.meta) as dst:
        for i in range(len(bands)):
            data_array = array_of_bands[i]
            data = np.digitize(data_array, breaks)
            dst.write(data, bands[i])
        dst.close()

    src = rasterio.open(path_to_file)

    return src

def raster_with_defined_interval(
    raster: rasterio.io.DatasetReader,
    path_to_file: str,
    interval_size,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
   
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    with rasterio.open(path_to_file, 'w', **raster.meta) as dst:
        for i in range(len(bands)):
            data_array = array_of_bands[i]
            print(interval_size)
            hist, edges = np.histogram(data_array, bins=interval_size)
            indices = np.digitize(data_array, edges)
            #bins = np.linspace(raster.statistics(i+1).min, raster.statistics(i+1).max + interval_size, num=interval_size, dtype='float64')
            #print(bins)
            #data = np.digitize(data_array, indices)
            dst.write(indices, bands[i])
        dst.close()

    src = rasterio.open(path_to_file)

    return src

def raster_with_equal_intervals(
    raster: rasterio.io.DatasetReader,
    path_to_file: str,
    number_of_intervals: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
   
    array_of_bands = []
    min_and_max_values = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    with rasterio.open(path_to_file, 'w', **raster.meta) as dst:
        for i in range(len(bands)):
            data_array = array_of_bands[i]
            max_value = raster.statistics(i+1).max
            min_value = raster.statistics(i+1).min
            percentiles = np.linspace(0, 100, number_of_intervals+1)
            bins = [min_value]
            for j in range(number_of_intervals):
                w = (max_value - np.min(bins)) / number_of_intervals
                bins.append(min_value + w * (j + 1))
            intervals = np.percentile(data_array, percentiles)
            print("ints",intervals)
            print("bins", bins)
            data = np.digitize(data_array, intervals)
            dst.write(data, bands[i])

        dst.close()

    src = rasterio.open(path_to_file)

    return src

def raster_with_quantiles(
    raster: rasterio.io.DatasetReader,
    path_to_file: str,
    number_of_quantiles: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
   
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    with rasterio.open(path_to_file, 'w', **raster.meta) as dst:
        for i in range(len(bands)):
            data_array = array_of_bands[i]
            
            intervals = [np.percentile(data_array, i * 100 / number_of_quantiles) for i in range(number_of_quantiles)]

            data = np.digitize(data_array, intervals)

            dst.write(data, bands[i])
        dst.close()

    src = rasterio.open(path_to_file)

    return src

def raster_with_natural_breaks(
    raster: rasterio.io.DatasetReader,
    path_to_file: str,
    number_of_breaks: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
   
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    with rasterio.open(path_to_file, 'w', **raster.meta) as dst:
        for i in range(len(bands)):
            data_array = array_of_bands[i]
            breaks = mc.JenksCaspall(data_array, number_of_breaks)
            data = np.digitize(data_array, np.sort(breaks.bins))

            dst.write(data, bands[i])
        dst.close()

    src = rasterio.open(path_to_file)

    return src

def raster_with_geometrical_intervals(
    raster: rasterio.io.DatasetReader,
    path_to_file: str,
    number_of_classes: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
   
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    with rasterio.open(path_to_file, 'w', **raster.meta) as dst:
        for i in range(len(bands)):
            data_array = array_of_bands[i]
            max_value = raster.statistics(i+1).max

            min_value = raster.statistics(i+1).min

            x = (max_value/np.int(min_value))**(1/number_of_classes+1)

            intervals = [min_value * x**i for i in range(1, number_of_classes+1)]

            intervals = np.array(intervals, dtype=np.float32)

            data = np.digitize(data_array, np.sort(intervals))
    
            dst.write(data, bands[i])
        dst.close()

    src = rasterio.open(path_to_file)

    return src

def raster_with_standard_deviation(
    raster: rasterio.io.DatasetReader,
    path_to_file: str,
    number_of_intervals: int,
    bands: Optional[List[int]] = None
) -> rasterio.io.DatasetReader:
   
    array_of_bands = []
    if bands is not None:
        for band in raster.read(bands):
            array_of_bands.append(band)
    else:
        array_of_bands = raster.read()

    with rasterio.open(path_to_file, 'w', **raster.meta) as dst:
        for i in range(len(bands)):

            data_array = array_of_bands[i]
            mean = raster.statistics(i+1).mean
 
            stddev = raster.statistics(i+1).std

            interval_size = stddev / number_of_intervals
            #intervals = []
            print(interval_size)
            print("raster.statistics(i+1).min", raster.statistics(i+1).min)
            intervals = [raster.statistics(i+1).min]
            for j in range(1, number_of_intervals+1):
                intervals.append(intervals[-1] + interval_size)

            print(intervals)
            print("raster.statistics(i+1).max", raster.statistics(i+1).max)
            # Apply the mask to the data
            #masked_data = np.ma.masked_array(data_array, intervals)
            data = np.digitize(data_array, np.sort(intervals))

            dst.write(data, bands[i])
        dst.close()

    src = rasterio.open(path_to_file)

    return src

def testi():
    # Get the number of intervals from the user
    num_intervals = int(input("Enter the number of intervals: "))

    # Set the NumPy array
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Calculate the standard deviation of the data
    std = np.std(data)

    # Calculate the interval size by dividing the standard deviation by the number of intervals
    interval_size = std / num_intervals
    print(interval_size)
    # Create the intervals by iterating over the data and adding the interval size to the previous value
    intervals = [data[0]]
    for i in range(1, len(data)):
        intervals.append(intervals[-1] + interval_size)

    # Classify the data using the intervals
    classified_data = np.zeros(data.shape)
    for i, val in enumerate(data):
        for j, interval in enumerate(intervals):
            if val >= interval:
                classified_data[i] = j

    print(classified_data)