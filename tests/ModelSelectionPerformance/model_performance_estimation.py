from eis_toolkit.ModelSelectionPerformance.Utilities.ConfigurationParameters import ConfigurationParameters
from eis_toolkit.ModelSelectionPerformance.DataSetLoader.DataSetLoader import DataSetLoader
from eis_toolkit.ModelSelectionPerformance.ModelSelection.inputSelectionAsFunction import MakeFeatureSelection

if __name__ == "__main__":
    
    """
     You can choose different cross validation methods going to the configuration file. 
     SKFOLD, LOO, KFOLD are the ones availlable. 
     The resulted model selecion performance is dispalyed as a coonfusion matrix    
    """


    # call of all confi parameters we need
    configuration_handler = ConfigurationParameters()

    # override configuration
    configuration_handler.configuration["cross_validation"] = True
    configuration_handler.configuration["epochs"] = 1
    configuration_handler.configuration["softmax"] = True
    configuration_handler.configuration["multiprocessing"] = False
    configuration_handler.configuration["cv_type"] = "SKFOLD"

    # load the dataset
    dataset_handler = DataSetLoader()
    dataset_handler.load_the_data_from_csv_file(path_to_gt=f"data/17_annoted_points.csv",
                                                path_to_point=f"data/2M_raster_points.csv")

    MakeFeatureSelection(dataset_handler=dataset_handler, configuration_handler=configuration_handler)

