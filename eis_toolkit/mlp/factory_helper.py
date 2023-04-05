import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

from eis_toolkit.exceptions import InvalidCrossValidationSelected, NumberOfSplitException


class factory_helper:
    """This is an helper class."""

    def __init__(self, general_configuration) -> None:
        """Class constructor."""
        self.general_configuration = general_configuration

    def cross_validation_methodology(self):
        """
        Do an instance of the cross validation method.

        Args:
            None.

        Returns:
            cross_validation_method (Object): Return the instance of
            the cross validation method.

        Raises:
            InvalidCrossValidationSelected: Invalid cross validation selected method.
        """

        if self.general_configuration["cv_type"] is None:
            raise InvalidCrossValidationSelected

        if self.general_configuration["number_of_split"] <= 1:
            raise NumberOfSplitException
        # stratified k fold
        if self.general_configuration["cv_type"] == "LOOCV":
            cross_validation_method = LeaveOneOut()

        if self.general_configuration["cv_type"] == "SKFOLD":
            cross_validation_method = StratifiedKFold(
                n_splits=self.general_configuration["number_of_split"], shuffle=True
            )

        if self.general_configuration["cv_type"] == "KFOLD":
            cross_validation_method = KFold(n_splits=self.general_configuration["number_of_split"], shuffle=True)

        return cross_validation_method

    def save_the_output_of_training(self, resulted_cv_folds: list) -> dict:
        """
          Save the resulted cross validation data.

        Args:
            Resulted_cv_folds (list): list of the resulted cv folds.

        Returns:
            Dictionary: with the following keys: mean_of_accuracy, concatenated_true_labels,
            concatenated_pred_labels, best_cv_round, best_round_accuracy.

        Raises:
            None.
        """
        handler = open(f'{self.general_configuration ["feature_save_folder"]}/accuracy_round.csv', "a")
        best_round_accuracy = list()
        sum = 0
        concatenated_true, concatenated_pred = None, None
        for ctn, element in enumerate(resulted_cv_folds):
            element[0].save(
                f'{self.general_configuration ["feature_save_folder"]}/my_model_fold_{ctn}_acc_{element[1]}.h5'
            )
            # write the acc
            handler.writelines(f"{element[1]},{element[2]}")
            sum += float(element[1])
            best_round_accuracy.append(element[1])

            np.save(f'{self.general_configuration ["feature_save_folder"]}/Y_true_{ctn}_my.npy', element[3])
            np.save(f'{self.general_configuration ["feature_save_folder"]}/y_pred_{ctn}_my.npy', element[4])

            # do the cm on the fly
            if concatenated_true is None:
                concatenated_true = np.array(element[3])
            else:
                concatenated_true = np.concatenate((concatenated_true, element[3]), axis=-1)

            if concatenated_pred is None:
                concatenated_pred = np.array(element[4])
            else:
                concatenated_pred = np.concatenate((concatenated_pred, element[4]), axis=-1)

        handler.close()

        return {
            "mean_of_accuracy": sum / float(len(resulted_cv_folds)),
            "concatenated_true_labels": concatenated_true,
            "concatenated_pred_labels": concatenated_pred,
            "best_cv_round": max(best_round_accuracy),
            "best_round_accuracy": best_round_accuracy.index(max(best_round_accuracy)),
        }

    def load_the_data_from_csv_file(self, path_to_load: str) -> np.array:
        """
        Load the data from the csv file.

        Args:
            path_to_load (str): path to the csv file.

        Returns:
            Numpy array: with the data and the labels.

        Raises:
            None.
        """

        data = pd.read_csv(path_to_load)
        label = data["class"]
        data = data.drop(["E", "N", "class"], axis=1)
        return data.to_numpy(), label.to_numpy()
