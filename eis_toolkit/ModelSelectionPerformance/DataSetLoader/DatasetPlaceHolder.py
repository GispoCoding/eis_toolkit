import numpy as np


class DataSetPlaceHolder:
    def __init__(self, raster_profile: tuple, data: np.ndarray, bands: int or list[int], types: str, path: str) -> None:
        """
        DtasetPlaceholder constructor
        :param raster_profile (str):
        :param data (np.ndarray):
        :param bands (int or list[tuple]):
        :param types (str): type of raster
        :param path (str): the path of the raster
        """
        self.raster_profile = raster_profile
        self.data = data
        self.bands = bands
        self.types = types
        self.path = path
        self.data_shape = self.__shape_checker()

    def __shape_checker(self):
        return self.data.shape

    def stringify(self):
        print(f"[PROFILE] Raster profile {self.raster_profile}")
        print(f"[DATA] Shape {self.data_shape}")
        print(f"[BANDS] Bands {self.bands}")
        print(f"[PATH] Path {self.path}")
