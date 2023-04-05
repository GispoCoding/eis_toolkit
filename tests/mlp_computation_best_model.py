import os

import numpy as np
import psutil
from sklearn.metrics import confusion_matrix

from eis_toolkit.mlp import mlp_factory

cnn_parameters_from_ht = {
    "input_channel_1": None,
    "dense_nodes": [16],
    "final_output": 2,
    "last_activation": "softmax",
    "l_2": 0.005,
    "rescaling": True,
}

configuration = {
    "cross_validation": True,
    "cv_type": "SKFOLD",
    "number_of_split": 5,
    "epochs": 500,
    "batch_size": 256,
    "number_of_csc_split": 400,
    "weight_samples": True,
    "verbose": 1,
    "feature_save_folder": "Results",
    "multiprocessing": True,
}

# create feat folder if does not exist
if not os.path.exists(f'{configuration["feature_save_folder"]}'):
    os.makedirs(f'{configuration["feature_save_folder"]}')

# istance of the mlp model
mlp_instance = mlp_factory.mlp_factory(
    cnn_configuration_dict=cnn_parameters_from_ht, general_configuration=configuration
)

# load the data
data_class_1, label_class_1 = mlp_instance.load_the_data_from_csv_file(
    path_to_load="data/local/data/17_annoted_points.csv"
)
data_class_0, label_class_0 = mlp_instance.load_the_data_from_csv_file(
    path_to_load="data/local/data/2M_raster_points.csv"
)

print(f"[CLASS 1] {data_class_1.shape}")
print(f"[CLASS 0] {data_class_0.shape}")
print(f"[MEMORY] Memory usage {psutil.virtual_memory()[2]} %")

# concatenate the data
concatenated_data = np.concatenate((data_class_1, data_class_0), axis=0)
concatenated_label = np.concatenate((label_class_1, label_class_0), axis=0)

encoded_concatenated_label = mlp_instance.make_the_one_hot_encoding(labels_to_transform=concatenated_label)

result_from_teaining = mlp_instance.do_the_fitting_with_microprocessing()


results = mlp_instance.prepare_weights_for_pixelwise_classification(
    X_train=concatenated_data, y_train=encoded_concatenated_label, weighted_samples=True
)

if configuration["verbose"] > 0:
    print(f"[ACCURACY] The mean of the accuracy is {results}")
    print(f"[CM] The CM is {confusion_matrix(y_true=results[3], y_pred=results[4])}")
