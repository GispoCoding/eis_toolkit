# Performing more complex tests
In case you do not want to insert your test commands one by one into the command line's python console, you can create a local test file and execute it with

```shell
python <name_of_your_test_file>.py
```

Your .py test file can, for example, look like:

```python
import rasterio as rio
import numpy as np
from matplotlib import pyplot
from pathlib import Path

output_path = Path('/home/pauliina/Downloads/eis_outputs/clip_result.tif')
src = rio.open(output_path)
arr = src.read(1)
# Let's replace No data values with numpy NaN values in order to plot clipped raster
# so that the colour changes are visible for human eye
arr = np.where(arr<-100, np.nan, arr)

pyplot.imshow(arr, cmap='gray')
pyplot.show()
```
