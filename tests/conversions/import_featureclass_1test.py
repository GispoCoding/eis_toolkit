
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit.conversions.import_featureclass import import_featureclass

# scripts = r"/eis_toolkit"  # /eis_toolkit/conversions'
# sys.path.append(scripts)


# input from GUI:
parent_dir = Path(__file__).parent.parent
name_fc = str(parent_dir.joinpath(r"data/shps/EIS_gp.gpkg"))
layer_name = r"Occ_2"
name_csv = parent_dir.joinpath(r"data/csv/Trainings_Test.csv")
name_wrong = str(parent_dir.joinpath(r"data/csv/Trainings_no.csv"))
# dfg = gpd.read_file(name_fc,layer = layer_name, driver = 'driver')
# columns and column-types for X (training based on a geopackage-layer)

fields_fc = {
    "OBJECTID": "i",
    "ID": "n",
    "Name": "n",
    "Alternativ": "n",
    "Easting_EU": "n",
    "Northing_E": "n",
    "Easting_YK": "n",
    "Northing_Y": "n",
    "Commodity": "c",
    "Size_by_co": "c",
    "All_main_c": "n",
    "Other_comm": "n",
    "Ocurrence_": "n",
    "Mine_statu": "n",
    "Discovery_": "n",
    "Commodity_": "n",
    "Resources_": "n",
    "Reserves_t": "n",
    "Calc_metho": "n",
    "Calc_year": "n",
    "Current_ho": "n",
    "Province": "n",
    "District": "n",
    "Deposit_ty": "n",
    "Host_rocks": "n",
    "Wall_rocks": "n",
    "Deposit_sh": "c",
    "Dip_azim": "v",
    "Dip": "t",
    "Plunge_azi": "v",
    "Plunge_dip": "v",
    "Length_m": "v",
    "Width_m": "n",
    "Thickness_": "v",
    "Depth_m": "n",
    "Date_updat": "n",
    "geometry": "g",
}

# columns and column-types for X (training based on a csv file)
fields_csv = {
    "LfdNr": "i",
    "Tgb": "t",
    "TgbNr": "n",
    "Inden_": "n",
    "SchneiderThiele": "c",
    "SuTNr": "c",
    "Asche_trocken": "v",
    "C_trocken": "n",
    "H_trocken": "v",
    "S_trocken": "v",
    "N_trocken": "v",
    "AS_Natrium_ppm": "v",
    "AS_Kalium_ppm": "v",
    "AS_Calcium_ppm": "v",
    "AS_Magnesium_ppm": "v",
    "AS_Aluminium_ppm": "v",
    "AS_Silicium_ppm": "v",
    "AS_Eisen_ppm": "v",
    "AS_Titan_ppm": "v",
    "AS_Schwefel_ppm": "n",
    "H_S": "v",
    "Na_H": "v",
    "K_H": "v",
    "Ca_H": "v",
    "Mg_H": "v",
    "Al_H": "v",
    "Si_H": "v",
    "Fe_H": "v",
    "Ti_H": "v",
    "Na_S": "v",
    "K_S": "v",
    "Ca_S": "v",
    "Mg_S": "v",
    "AL_S": "v",
    "Si_S": "v",
    "Fe_S": "v",
    "Ti_S": "v",
    "Na_K": "v",
    "Na_Ca": "v",
    "Na_Mg": "v",
    "Na_Al": "v",
    "Si_Na": "v",
    "Na_Fe": "v",
    "Na_Ti": "v",
    "K_Ca": "v",
    "K_Mg": "v",
    "K_Al": "v",
    "Si_K": "v",
    "K_Fe": "v",
    "K_Ti": "v",
    "Ca_Mg": "v",
    "Ca_Al": "v",
    "Si_Ca": "v",
    "Ca_Fe": "v",
    "Ca_Ti": "v",
    "Mg_Al": "v",
    "Si_Mg": "v",
    "Mg_Fe": "v",
    "Mg_Ti": "v",
    "Si_Al": "v",
    "Al_Fe": "v",
    "Al_Ti": "v",
    "Si_Fe": "v",
    "Si_Ti": "v",
    "Fe_Ti": "v",
}


def test_import_featureclass_fc():
    """Test functionality import of fc layer."""
    columns, df, urdf, metadata = import_featureclass(fields=fields_csv, file=name_csv, layer=layer_name)

    assert isinstance(columns, dict)
    assert isinstance(df, (gpd.GeoDataFrame, pd.DataFrame))
    assert isinstance(urdf, (gpd.GeoDataFrame, pd.DataFrame))
    assert len(columns) > 0
    assert len(df.index) > 0
    assert len(df.columns) > 0


def test_import_featureclass_csv():
    """Test functionality import of fc layer."""
    columns, df, urdf, metadata = import_featureclass(fields=fields_csv, file=name_csv, decimalpoint_german=True)

    assert isinstance(columns, dict)
    assert isinstance(df, (gpd.GeoDataFrame, pd.DataFrame))
    assert isinstance(urdf, (gpd.GeoDataFrame, pd.DataFrame))
    assert len(columns) > 0
    assert len(df.index) > 0
    assert len(df.columns) > 0


def test_import_featureclass_wrong():
    """Test functionality import with wrong file name."""
    with pytest.raises(BeartypeCallHintParamViolation):  # InvalidParameterValueException):
        columns, df, urdf, metadata = import_featureclass(fields=fields_csv, file=name_wrong, decimalpoint_german=6)


# test_import_featureclass_fc()
test_import_featureclass_csv()
test_import_featureclass_wrong()
