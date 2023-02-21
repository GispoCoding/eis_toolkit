from MLP import *


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


# Plot confusion matrix using seaborn
def make_cm_plot_HARCODED_TO_REMOVE_LATER(cm, true_labels, pred_labels, round_number, neurons_number, l_2_value):
    """
    This is the internal function that construct the confusion matrix plot
    """
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Reds', fmt='g')
    accuracy = accuracy_score(y_true=np.array(true_labels), y_pred=np.array(pred_labels))
    accuracy = round (accuracy *100,2)
    plt.xlabel(f'Predicted Label')
    #plt.xlabel('Accuracy = {:.2f}% '.format(accuracy*100))
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix Accuracy: {accuracy}% l2: {l_2_value}')
    #plt.show()
    plt.savefig(f'cm_round_{round_number}_nn_{neurons_number}_l2_{l_2_value}.png')
    plt.close()


#load the data 
data_class_1, label_class_1 = load_the_data_from_csv_file(path=f"/home/dipak/Desktop/Final_EIS_Data/17_annoted_points.csv")
data_class_0, label_class_0 = load_the_data_from_csv_file(path=f"/home/dipak/Desktop/Final_EIS_Data/2M_raster_points.csv")

print(f'[CLASS 1] {data_class_1.shape}')
print(f'[CLASS 0] {data_class_0.shape}')

#list of params 
list_of_params_we_need = {
    'data_class_0':data_class_0, 
    'data_class_1': data_class_1, 
    'label_class_0': label_class_0,
    'label_class_1': label_class_1, 
    'write_report':"report.csv",
    'list_of_hidden_layers': [2], 
    'pixel_normalization': False ,
    'weight_samples':True, 
    'epochs': 1, 
    'batch_size': 2, 
    'cm_round': True
}

MLP = MLP(list_of_params_we_need)
returned_dictionary = MLP.compute_MLP_workload(**list_of_params_we_need)


print(returned_dictionary)