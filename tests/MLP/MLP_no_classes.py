import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import numpy as np
import pandas as pd
import math
import random


class MLP:
    def __init__(self, mlp_prameters: dict):
        self.mlp_parameters = mlp_prameters



def make_the_one_hot_encoding(labels_to_transform):
    try:
        """
           We need to make one hot encoding because we have used softmax activation for 2 classes
        """
        enc = OneHotEncoder(handle_unknown='ignore')
        # this is a trick to figure the array as 2d array instead of list
        temp = np.reshape(labels_to_transform, (-1, 1))
        labels_to_transform = enc.fit_transform(temp).toarray()
        # print(f'[ONE HOT ENCODING] Labels are one-hot-encoded: {(labels_to_transform.sum(axis=1) - np.ones(labels_to_transform.shape[0])).sum() == 0}')
        return labels_to_transform
    except Exception as ex:
        print(f"[EXCEPTION] Make the one hot encoding throws exception {ex}")


def generate_dynamically_the_MLP(input_channel_1: tuple, rescaling: bool ,dense_nodes: list, final_output: int, last_activation: str, l_2: float):
    """
        This function generates Multilayer perceptron based on the parameters, where input channel is the tuple, dense_node is number of neural node in the hidden layer, final_output is the number of labels,
        last_activation is the type of activation function at the last layer, and l_2 is the L2 regularization value given to the network.
        This function returns the generated model.
    """
    input_layer = tf.keras.Input(shape=input_channel_1, name="input_1")

    if rescaling:
        body = tf.keras.layers.Rescaling(1.0 / 255)(input_layer)
        # one conv 1d example
        # we need flatten
        flatten = tf.keras.layers.Flatten(name=f"flatten_layer")(body)
    else:
        flatten = tf.keras.layers.Flatten(name=f"flatten_layer")(input_layer)


    if len(dense_nodes) > 0:
        # add the classifier
        for ctn, dense_unit in enumerate(dense_nodes):
            if ctn == 0:
                classifier = tf.keras.layers.Dense(dense_unit,
                                                   activation='relu',
                                                   kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=l_2))(flatten)
            else:
                classifier = tf.keras.layers.Dense(dense_unit,
                                                   activation='relu',
                                                   kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=l_2))(classifier)

                # add the final output
        classifier = tf.keras.layers.Dense(final_output,
                                           activation=last_activation,
                                           kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                           bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                           name=f"final_classifier")(classifier)
    else:
        # add the final output
        classifier = tf.keras.layers.Dense(final_output,
                                           activation=last_activation,
                                           kernel_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                           bias_regularizer=tf.keras.regularizers.L2(l2=l_2),
                                           name=f"final_classifier")(flatten)
        # create the model obj
    model = tf.keras.Model(inputs=input_layer, outputs=classifier, name=f"model_1")
    return model


def compute_MLP_workload(data_class_0, data_class_1, label_class_0, label_class_1, write_report="report.csv",
                list_of_hidden_layers=[2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 50, 64], pixel_normalization=False ,
                weight_samples=True, epochs=100, batch_size=2, cm_round=True):
    """
    This function compute the multilayer perceptron value based on the parameters given to it.
    It takes the data as input for class 0 and class 1 along with their labels.
    write_report writes the summary of the model in .csv file during training.
    list_of_hidden_layers is the parameters for the number of neurons in the hidden layer.
    pixel_normalization to False means that no pixel normalization will be applied to the image
    Weight_samples = True, can be a useful technique for improving the performance of a model, especially when dealing with imbalanced datasets.
    The number of epochs is a hyperparameter that needs to be specified by the user and represents the number of times the model will be trained on the entire dataset.
    cm_round: calculate the confusion matrix for the rounds
    This function returns the true_label_list, predicted_label_list, score, hidden_layer neurons, L2 regularization value , confusion_matrix and train_history

    """
    try:
        if write_report is not None:
            handler = open(f'{write_report}', 'w')

        number_of_true_samples = len(data_class_1)

        # number of total combination
        number_of_total_combination = math.floor(data_class_0.shape[0] / number_of_true_samples)

        # I need this to safeget stuff from the memory
        list_of_already_taken_id = list()

        # make a big list that for each rounds include a list of true and pred elements
        returned_dict = {}

        for round_number in range(number_of_total_combination + 1):
            # get the random number
            class_0_we_need, label_0_we_need = list(), list()

            list_of_index = list()

            if len(list_of_already_taken_id) + number_of_true_samples > len(data_class_0):
                print(f"[SAMPLES AVAILABILITY] No more samples availlable")

                for ii in range(len(list_of_already_taken_id), len(data_class_0)):
                    list_of_index.append(ii)

            else:

                while len(list_of_index) < number_of_true_samples:
                    random_number = random.randint(0, len(data_class_0) - 1)

                    # if the number is not in the list add to the random numbers
                    if random_number not in list_of_already_taken_id:
                        list_of_already_taken_id.append(random_number)
                        # print(f"[AVAILLABLE INDEX] {random_number}")
                        list_of_index.append(random_number)
                    else:
                        continue

            for index in list_of_index:
                class_0_we_need.append(data_class_0[index, :])
                label_0_we_need.append(label_class_0[index])

            class_0_we_need = np.array(class_0_we_need)
            label_0_we_need = np.array(label_0_we_need)

            data = np.concatenate((data_class_1, class_0_we_need), axis=0)
            labels = np.concatenate((label_class_1, label_0_we_need), axis=0)

            # we make 1 hot encoding for the labels
            labels = make_the_one_hot_encoding(labels_to_transform=labels)

            # loocv (CV
            loo = LeaveOneOut()
            num_of_split = loo.get_n_splits(data)
            print(f'[LOOCV] Number of split is {num_of_split} start the cross validation')

            for hidden_layer in list_of_hidden_layers:
                for l2_current_values in np.logspace(-1, 1, num=10, base=10):

                    # create the list for the cm
                    true_label_list, predicted_laberl_list = [], []

                    print(f'[CURRENT] Hidden layer {hidden_layer} and L2 is {l2_current_values}')
                    round_score_list = list()
                    for ii, (train_index, test_index) in enumerate(loo.split(data)):

                        X_train, X_test = data[train_index], data[test_index]
                        y_train, y_test = labels[train_index], labels[test_index]

                        # reshape the data to get number of sample 1 row of 20 columns
                        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                        # instance of the model and compile
                        model = generate_dynamically_the_MLP(input_channel_1=(X_train.shape[1], X_train.shape[2]),
                                                            rescaling=pixel_normalization,
                                                            dense_nodes=[hidden_layer],
                                                            final_output=labels.shape[1],
                                                            last_activation='softmax',
                                                            l_2=l2_current_values)
                        # model.summary()
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                        if weight_samples:
                            # create class weights
                            samples_weights = compute_sample_weight("balanced", y_train)

                            # train
                            history = model.fit(X_train, y_train,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                validation_data=(X_test, y_test),
                                                verbose=0,
                                                sample_weight=samples_weights)
                        else:

                            history = model.fit(X_train, y_train,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                validation_data=(X_test, y_test),
                                                verbose=0)


                        # eval on loocv
                        score = model.evaluate(X_test, y_test, verbose=0)

                        # add the round score
                        round_score_list.append(score[1])

                        # predict
                        prediction = model.predict(X_test, verbose=0)

                        predicted_laberl_list.append(np.argmax(prediction))
                        true_label_list.append(np.argmax(y_test))
                        
                    if write_report is not None:
                        # add the mean of acc
                        handler.writelines(f'{round_number},{hidden_layer},{np.mean(round_score_list)}\n')

                    if cm_round:
                        # do the cm for the rounds
                        round_cm = confusion_matrix(y_true=np.array(true_label_list), y_pred=np.array(predicted_laberl_list))
                    
                    # prepare the dict to return 
                    returned_dict[f'Round_{round_number}'] = {'True': true_label_list, 'Pred': predicted_laberl_list, 'Score': round_score_list, 'Hidden_layer': hidden_layer, 'L2': l2_current_values, 'CM': round_cm, 'train_history': history.history}

        # if a stream is opened close it 
        if write_report is not None:
            handler.close()
        
        return returned_dict 

    except Exception as ex:
        print(f"Main throws exception {ex}")