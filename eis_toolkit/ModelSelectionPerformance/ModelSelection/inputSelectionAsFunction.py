from sklearn.metrics import confusion_matrix
from eis_toolkit.ModelSelectionPerformance.NNUtilities.NNFactory import NNFactory
from eis_toolkit.ModelSelectionPerformance.ModelPerformanceEstimation.ModelPerformanceEstimation import ModelPerformanceEstimation
from eis_toolkit.ModelSelectionPerformance.DataSetLoader.DataSetLoader import DataSetLoader
from joblib import Parallel, delayed
from  eis_toolkit.ModelSelectionPerformance.Utilities.ConfigurationParameters import ConfigurationParameters
import numpy as np 
import os


def MakeFeatureSelection(dataset_handler: DataSetLoader, configuration_handler: ConfigurationParameters) -> None:
    """
    Do the Model Selection and return the performance of cross validation
    :param dataset_handler (DtasetLoader): A class used to manage the data
    :param configuration_handler (ConfigurationParameters) A class used to stack all configuration necessary for running:
    :return: None
    """

    handler = NNFactory(configuration_handler=configuration_handler)

    # create dirs
    if not os.path.exists(f'{configuration_handler.configuration["feature_save_folder"]}'):
        os.makedirs(f'{configuration_handler.configuration["feature_save_folder"]}')

    selected_cs = ModelPerformanceEstimation(cross_validation_type=configuration_handler.configuration["cv_type"],
                                             number_of_split=configuration_handler.configuration["number_of_split"])

    if configuration_handler.configuration["multiprocessing"]:
        # do cv multiproc without sample weight
        resulted_cv = Parallel(n_jobs=int(configuration_handler.configuration["number_of_split"]),
                               backend='multiprocessing')(
            delayed(handler.do_the_fitting_with_microprocessing)(worker=counter,
                                                                 X_train=dataset_handler.current_dataset[train_idx],
                                                                 y_train=dataset_handler.encoded_labels[train_idx] if
                                                                 configuration_handler.configuration['softmax'] else
                                                                 dataset_handler.current_labels[train_idx],
                                                                 X_test=dataset_handler.current_dataset[validation_idx],
                                                                 y_test=dataset_handler.encoded_labels[
                                                                     validation_idx] if
                                                                 configuration_handler.configuration['softmax'] else
                                                                 dataset_handler.current_labels[validation_idx]
                                                                 ) for counter, (train_idx, validation_idx)
            in enumerate(selected_cs.cross_validation_method.split(np.zeros(len(dataset_handler.current_dataset)),
                                                                   dataset_handler.current_labels)))
    else:
        # get the mean of acc
        resulted_cv = list()
        # sample weighted
        for ii, (train_idx, validation_idx) in enumerate(
                selected_cs.cross_validation_method.split(np.zeros(len(dataset_handler.current_dataset)),
                                                          dataset_handler.current_labels)):
            # the folds
            result = handler.do_the_fitting_with_microprocessing(worker=ii,
                                                                 X_train=dataset_handler.current_dataset[train_idx],
                                                                 y_train=dataset_handler.encoded_labels[train_idx] if
                                                                 configuration_handler.configuration['softmax'] else
                                                                 dataset_handler.current_labels[train_idx],
                                                                 X_test=dataset_handler.current_dataset[validation_idx],
                                                                 y_test=dataset_handler.encoded_labels[
                                                                     validation_idx] if
                                                                 configuration_handler.configuration['softmax'] else
                                                                 dataset_handler.current_labels[validation_idx])

            resulted_cv.append(result)

    stacked_true, stacked_pred = None, None
    average_of_eval = 0
    for counter, element in enumerate(resulted_cv):
        # add the sum to do the avg of the model
        average_of_eval += element[1]

        if stacked_true is None:
            stacked_true = element[3]
        else:
            stacked_true = np.concatenate((stacked_true, element[3]))

        if stacked_pred is None:
            stacked_pred = element[4]
        else:
            stacked_pred = np.concatenate((stacked_pred, element[4]))

        cm = confusion_matrix(stacked_true, stacked_pred)
        df = pd.DataFrame(cm, columns=[sorted(np.unique(dataset_handler.current_labels))],
                          index=sorted(np.unique(dataset_handler.current_labels)))

        if configuration_handler.configuration["save"]:
            # save the ds
            df.to_csv(f"cm_accuracy_{average_of_eval / configuration_handler.configuration['number_of_split']}.csv")

        print(f"[AVERAGE ACCURACY] The accuracy average is {average_of_eval / configuration_handler.configuration['number_of_split']}")




