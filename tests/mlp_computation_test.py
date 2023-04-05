import os

import numpy as np
import psutil
from joblib import Parallel, delayed
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

# one hot encoded data
encoded_concatenated_label = mlp_instance.make_the_one_hot_encoding(labels_to_transform=concatenated_label)
cross_validation_method = mlp_instance.cross_validation_methodology()

if configuration["multiprocessing"]:
    # do cv multiproc without sample weight
    resulted_cv = Parallel(n_jobs=int(configuration["number_of_split"]), backend="multiprocessing")(
        delayed(mlp_instance.do_the_fitting_with_microprocessing)(
            worker=counter,
            X_train=concatenated_data[train_idx],
            y_train=encoded_concatenated_label[train_idx],
            X_test=concatenated_data[validation_idx],
            y_test=encoded_concatenated_label[validation_idx],
            weighted_samples=configuration["weight_samples"],
        )
        for counter, (train_idx, validation_idx) in enumerate(
            cross_validation_method.split(np.zeros(len(concatenated_data)), concatenated_label)
        )
    )

    dictionary_from_cv = mlp_instance.save_the_output_of_training(resulted_cv_folds=resulted_cv)

    cm = confusion_matrix(
        y_true=dictionary_from_cv["concatenated_true_labels"], y_pred=dictionary_from_cv["concatenated_pred_labels"]
    )

    if configuration["verbose"] > 0:
        print(f"[ACCURACY] The mean of the accuracy is {dictionary_from_cv}")
        print(f"[CM] The CM is {cm}")

else:
    # get the mean of acc
    mean_of_accuracy = list()
    temp_list_hold_fold = list()

    # sample weighted
    for ii, (train_idx, validation_idx) in enumerate(
        cross_validation_method.split(np.zeros(len(concatenated_data)), concatenated_label)
    ):
        # the folds
        result = mlp_instance.do_the_fitting_with_microprocessing(
            worker=ii,
            cnn_configuration_dict=cnn_parameters_from_ht,
            X_train=concatenated_data[train_idx],
            y_train=encoded_concatenated_label[train_idx],
            X_test=concatenated_data[validation_idx],
            y_test=encoded_concatenated_label[validation_idx],
            weighted_samples=configuration["weight_samples"],
        )

        temp_list_hold_fold.append(result)

    dictionary_from_cv = mlp_instance.save_the_output_of_training(resulted_cv_folds=temp_list_hold_fold)

    cm = confusion_matrix(
        y_true=dictionary_from_cv["concatenated_true_labels"], y_pred=dictionary_from_cv["concatenated_pred_labels"]
    )

    if configuration["verbose"] > 0:
        print(f"[ACCURACY] The mean of the accuracy is {dictionary_from_cv}")
        print(f"[CM] The CM is {cm}")
