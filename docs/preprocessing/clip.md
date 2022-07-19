#


### clip_ras
[source](https://github.com/GispoCoding/eis_toolkit/blob/master/eis_toolkit/preprocessing/clip.py/#L11)
```python
.clip_ras(
   rasin: Path, polin: Path
)
```

---
Clips raster with polygon.


**Args**

* **rasin** (Path) : file path to input raster
* **polin** (Path) : file path to polygon to be used for clipping the input raster


**Returns**

* tuple consisting of clipped raster in array format and georeferencing information
