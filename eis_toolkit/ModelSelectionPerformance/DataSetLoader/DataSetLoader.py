import os
import pandas as pd
import numpy as np
import psutil
from osgeo import gdal
from eis_toolkit.exceptions import CanNotMakeCategoricalLabelException, NoSuchPathOrDirectory
from sklearn.preprocessing import OneHotEncoder
from eis_toolkit.ModelSelectionPerformance.DataSetLoader.DatasetPlaceHolder import DataSetPlaceHolder

class DataSetLoader:
    def __init__(self, path_to_raster: str = None) -> None:
        """
        Class constructor
        :param path_to_raster (str): Path to the folder of rastrt
        """
        self.path_to_raster = path_to_raster
        self.current_dataset = None
        self.current_labels = None
        self.encoded_labels = None

    def __csv_loader(self, path):
        """
        This function reads the data from the path and convert it into dataframe.
        labels are seperated from data.
        Data columns containing name [’E’,'N','class'] are dropped.
        This function returns data and labels into numpy array.
        """
        data = pd.read_csv(path)
        label = data['class']
        data = data.drop(['E', 'N', 'class'], axis=1)
        return data.to_numpy(), label.to_numpy()

    def __make_the_one_hot_encoding(self):
        """
            you need to make one hot encoding because we use the stragety 2 classes and softmax activation
        """
        enc = OneHotEncoder(handle_unknown='ignore')
        temp = np.reshape(self.current_labels, (-1, 1))
        self.encoded_labels = enc.fit_transform(temp).toarray()

        if (self.encoded_labels.sum(axis=1) - np.ones(self.encoded_labels.shape[0])).sum() != 0:
            raise CanNotMakeCategoricalLabelException

    def load_the_data_from_csv_file(self, path_to_gt: str, path_to_point: str) -> None:
        """
        Load the data from csv file
        :param path_to_gt (str): path to the gt csv file
        :param path_to_point (str): path to the other points
        """
        # load the data
        data_class_1, label_class_1 = self.__csv_loader(path=f"{path_to_gt}")
        data_class_0, label_class_0 = self.__csv_loader(path=f"{path_to_point}")

        print(f'[CLASS 1] {data_class_1.shape}')
        print(f'[CLASS 0] {data_class_0.shape}')
        print(f'[MEMORY] Memory usage {psutil.virtual_memory()[2]} %')

        # concatenate the data
        self.current_dataset = np.concatenate((data_class_1, data_class_0), axis=0)
        self.current_labels = np.concatenate((label_class_1, label_class_0), axis=0)

        # normalize ds
        self.current_dataset = self.current_dataset / 255.

        # create one hot
        self.__make_the_one_hot_encoding()

    def load_the_data_from_raster(self) -> None:
        """
            Create dataset from raster
            It creates an object from that ()
            :return: (list[objects]) DatasetPlaceholder
        """
        # current ds became list
        self.current_dataset = list()
        # check if path exist
        if not os.path.exists(self.path_to_raster):
            raise NoSuchPathOrDirectory

        # walk in all the paths
        for path in os.listdir(f'{self.path_to_raster}'):
            satellite_path = f"{self.path_to_raster}/{path}"
            for tif_file in os.listdir(f'{satellite_path}'):
                # get the full path of satellite
                full_raster_path = f"{satellite_path}/{tif_file}"
                filename, extension = os.path.splitext(full_raster_path)

                # exclude the non tif file
                if extension != ".tif":
                    continue

                print(f"[WALK] walking in {full_raster_path}")

                # load the image
                tiff_data = gdal.Open(f"{full_raster_path}", gdal.GA_ReadOnly)

                # get the tif profile
                geoprofile = tiff_data.GetGeoTransform()

                minx = geoprofile[0]  # x-koordinaatin minimi (lansi) W
                maxy = geoprofile[3]  # y-koordinaatin maksimi (pohjoinen) N
                pix_x = geoprofile[1]  # pikselikoko x-suunnassa; positiivinen (kasvaa lanteen)
                pix_y = geoprofile[5]  # pikselikoko y-suunnassa; negatiivinen (pienenee etelaan)
                x_ext = tiff_data.RasterXSize  # rasterin koko (pikselia) x-suunnassa
                y_ext = tiff_data.RasterYSize  # rasterin koko (pikselia) y-suunnassa
                maxx = minx + pix_x * x_ext  # x-koordinaatin maksimi (ita) E
                miny = maxy + pix_y * y_ext  # y-koordinaatin minimi (etela) S

                print(f"[PROFILE] image: {full_raster_path} has profile:\n"
                      f"[min_x] {minx}\n"
                      f"[max_y] {maxy}\n"
                      f"[pix_x] {pix_x}\n"
                      f"[pix_y] {pix_y}\n"
                      f"[x_ext] {x_ext}\n"
                      f"[y_ext] {y_ext}\n"
                      f"[max_x] {maxx}\n"
                      f"[min_y] {miny}\n")

                self.current_dataset.append(
                    DataSetPlaceHolder(raster_profile={
                        "min_x": minx,
                        "max_x": maxx,
                        "min_y": miny,
                        "max_y": maxy,
                        "pix_x": pix_x,
                        "pix_y": pix_y,
                        "high_in_meters": tiff_data.RasterYSize,
                        "width_in_meters": tiff_data.RasterYSize
                    },
                                       data=tiff_data.ReadAsArray(),
                                       bands=tiff_data.RasterCount,
                                       types=satellite_path,
                                       path=full_raster_path)
                )

"""
if __name__ == '__main__':
    try:
        ds_handler = DataSetLoader(path_to_raster="../Annotations/Geophysical_Data")
        ds_handler.load_the_data_from_raster()
    except Exception as ex:
        print(f"[EXCEPTION] Main throws exception {ex}")
"""


