#from Configuration.configuration_parameters import *
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from exceptions import *
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import matplotlib.pyplot as plt


def do_the_scaled_ds(path, number_of_column_I_need):
    try:
        ds = pd.read_csv(path)
        # get cols
        E = ds['E']
        N = ds['N']
        c = ds['class']

        # drop the columns because we do no want to scale E N and the classes
        ds = ds.drop(['E', 'N', 'class'], axis=1)
        scaled = MinMaxScaler().fit_transform(np.array(ds))
        scaled_ds = pd.DataFrame(scaled)
        scaled_ds.columns = ds.columns[0:number_of_column_I_need]

        # add again the column
        scaled_ds['E'] = E
        scaled_ds['N'] = N
        scaled_ds['class'] = c
        #
        return scaled_ds
    except Exception as ex:
        print(f"[EXCEPTION]  Do the scaled ds throws exception {ex}")


def save_the_output_of_training(result, configuration):
    handler = open(f'{configuration["feature_save_folder"]}/accuracy_round.csv', 'a')
    sum = 0
    for ctn, element in enumerate(result):
        # get the model of the round
        element[0].save(f'{configuration["feature_save_folder"]}/my_model_fold_{ctn}_acc_{element[1]}.h5')
        # write the acc
        handler.writelines(f"{element[1]},{element[2]}")
        # make the sum of acc
        sum += float(element[1])
        # arrays for cms
        np.save(f'{configuration["feature_save_folder"]}/Y_true_{ctn}_my.npy', element[3])
        np.save(f'{configuration["feature_save_folder"]}/y_pred_{ctn}_my.npy', element[4])
    handler.close()
    # make mean and return
    return sum / float(len(result))


def do_the_score_and_predition(configuration, model, y_test, x_test, fold, model_name):
    try:
        predictions = model.predict(x_test)
        np.save(f'Results/y_true_{fold}_{model_name}.npy', y_test)
        np.save(f'Results/y_pred_{fold}_{model_name}.npy', predictions)
        joblib.dump(model, f"{configuration['feature_save_folder']}/{model_name}_fold_{fold}.joblib")
        fold_score = model.score(x_test, y_test)
        print(f"[FOLD SCORE] The fold score for the model {model_name} is {fold_score}")
        return fold_score
    except Exception as ex:
        print(f"[EXCEPTION] Do the score and predictions throws exception {ex}")

def do_the_hyper_parameters_tuning(types):
    try:
        if types == "rf":
            n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            return random_grid
    except Exception as ex:
        print(f"[EXCEPTION] do hyper parameters tuning throws exception {ex}")


def load_the_data_from_csv_file(path):
    """
    This function reads the data from the path and convert it into dataframe.
    labels are seperated from data.
    Data columns containing name [’E’,'N','class'] are dropped.
    This function returns data and labels into numpy array.
    """
    try:
        data = pd.read_csv(path)
        label = data['class']
        data = data.drop(['E', 'N', 'class'], axis=1)
        return data.to_numpy(), label.to_numpy()

    except Exception as ex:
        print(ex)
        return None


def make_the_one_hot_encoding(labels_to_transform):
    """
        you need to make one hot encoding because we use the stragety 2 classes and softmax activation
    """
    enc = OneHotEncoder(handle_unknown='ignore')
    temp = np.reshape(labels_to_transform, (-1, 1))
    labels_to_transform = enc.fit_transform(temp).toarray()

    if (labels_to_transform.sum(axis=1) - np.ones(labels_to_transform.shape[0])).sum() != 0:
        raise CanNotMakeCategoricalLabelException

    return labels_to_transform


def cross_validation_methodology(cross_validation_type, number_of_split):
    if cross_validation_type is None:
        raise InvalidCrossValidationSelected

    if number_of_split <= 1:
        raise NumberOfSplitException
    # stratified k fold
    if cross_validation_type == 'LOOCV':
        cross_validation_method = LeaveOneOut()

    if cross_validation_type == 'SKFOLD':
        cross_validation_method = StratifiedKFold(n_splits=number_of_split, shuffle=True)

    if cross_validation_type == 'KFOLD':
        cross_validation_method = KFold(n_splits=number_of_split, shuffle=True)

    return cross_validation_method


def do_the_chart(result, list_of_index, save_filename, title):
    # make the series
    imp = pd.Series(result, index=list_of_index).sort_values(ascending=True)
    # plot the series
    ax = imp.plot.barh()
    ax.set_title(f"{title}")
    ax.figure.tight_layout()
    # plt.show()
    plt.savefig(f'{save_filename}')


def prepare_the_data(path_class_0, path_class_1):
    try:
        dataset = {}
        class_0 = pd.read_csv(f"{path_class_0}")
        class_1 = pd.read_csv(f"{path_class_1}")

        # get the coordinates of class 0
        dataset['class_0_E'] = np.array(class_0['E'])
        dataset['class_0_N'] = np.array(class_0['N'])
        dataset['class_1_E'] = np.array(class_1['E'])
        dataset['class_1_N'] = np.array(class_1['N'])

        # combine ds
        temp = pd.concat([class_1.drop(['N', 'E'], axis=1), class_0.drop(['N', 'E'], axis=1)])
        dataset['label'] = np.array(temp['class'])

        # drop class
        temp = temp.drop(['class'], axis=1)

        # prepare the data and the data scaled
        dataset['data'] = temp.to_numpy(dtype='float')
        dataset['data_scaled'] = MinMaxScaler().fit_transform(dataset['data'])
        dataset['data_normalized'] = StandardScaler().fit_transform(dataset['data'])

        return dataset
    except Exception as ex:
        print(f"[EXCEPTION] Main throws exception {ex}")