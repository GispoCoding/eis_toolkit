

class ConfigurationParameters():
    """
        All configuration parameters needed for MLP, CNN
    """
    def __init__(self):
        self.cnn_parameters_from_ht = {
            "input_channel_1": None,
            "dense_nodes": [16, 32],
            "final_output": 2,
            "last_activation": "softmax",
            "l_2": 0.005,
            "rescaling": False
        }

        self.configuration = {
            "cross_validation": True,
            "cv_type": "SKFOLD",
            "number_of_split": 5,
            "epochs": 500,
            "batch_size": 256,
            "number_of_csc_split": 400,
            "weight_samples": True,
            "verbose": True,
            "feature_save_folder": "Results",
            "multiprocessing": True,
            "softmax": True,
            "other_model": False,
            "save": False
        }

        self.annotations = {
            "AEM": "../Annotations/Geophysical_Data/AEM",
            "Gravity": "../Annotations/Geophysical_Data/Gravity",
            "Magnetic": "../Annotations/Geophysical_Data/Magnetic",
            "Radiometric": "../Annotations/Geophysical_Data/Radiometric"
        }

        self.pixelwise_classification_options = {
            "OVERRIDE_RASTER_SIZE": 50
        }

        self.list_of_index = ["Mag_TMI", "Mag_AS", "DRC135", "DRC180", "DRC45", "DRC90", "Mag_TD", "HDTDR", "Mag_Xdrv",
                              "mag_Ydrv", "Mag_Zdrv", "Pseu_Grv", "Rd_U", "Rd_TC", "Rd_Th", "Rd_K", "EM_ratio", "EM_Ap_rs",
                              "Em_Qd", "EM_Inph"]

        # change the output type if there is sigmoid
        if self.configuration["softmax"] is False:
            self.cnn_parameters_from_ht["final_output"] = 1
            self.cnn_parameters_from_ht["last_activation"] = 'sigmoid'

def return_all_needed_parameters():
    """
    This function return one time all the necessary default config
    Return:
        configuration (dict)
        cnn_parameters_from_ht (dict)
        annotations (dict)
        pixelwise_classification_options (dict)
        list_of_index (list) list of index
    """

    cnn_parameters_from_ht = {
        "input_channel_1": None,
        "dense_nodes": [16, 32],
        "final_output": 1,
        "last_activation": "sigmoid",
        "l_2": 0.005,
        "rescaling": False
    }

    configuration = {
        "cross_validation": False,
        "cv_type": "SKFOLD",
        "number_of_split": 5,
        "epochs": 500,
        "batch_size": 256,
        "number_of_csc_split": 400,
        "weight_samples": True,
        "verbose": True,
        "feature_save_folder": "Results",
        "multiprocessing": False,
        "softmax": False,
        "other_model": False
    }

    annotations = {
        "AEM": "../Annotations/Geophysical_Data/AEM",
        "Gravity": "../Annotations/Geophysical_Data/Gravity",
        "Magnetic": "../Annotations/Geophysical_Data/Magnetic",
        "Radiometric": "../Annotations/Geophysical_Data/Radiometric"
    }

    pixelwise_classification_options = {
        "OVERRIDE_RASTER_SIZE": 50
    }

    list_of_index =["Mag_TMI", "Mag_AS", "DRC135", "DRC180", "DRC45", "DRC90", "Mag_TD", "HDTDR", "Mag_Xdrv", "mag_Ydrv",
                    "Mag_Zdrv", "Pseu_Grv", "Rd_U", "Rd_TC", "Rd_Th", "Rd_K", "EM_ratio", "EM_Ap_rs", "Em_Qd", "EM_Inph"]

    # change the output type if there is sigmoid
    if configuration["softmax"] is False:
        cnn_parameters_from_ht["final_output"] = 1
        cnn_parameters_from_ht["last_activation"] = 'sigmoid'

    return configuration, cnn_parameters_from_ht, annotations, pixelwise_classification_options, list_of_index