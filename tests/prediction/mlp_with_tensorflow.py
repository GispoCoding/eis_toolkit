from eis_toolkit.prediction.cnn_mlp_tensorflow import do_training_and_prediction_of_the_model

if __name__ == '__main__':
    df, model_to_return = do_training_and_prediction_of_the_model(deposit_path="/media/luca/T7 Shield/Eis_data/Annotations/17_annoted_points.csv",
                                                                  unlabelled_data_path="/media/luca/T7 Shield/Eis_data/Annotations/2M_raster_points.csv",
                                                                  path_to_features="masterfile_eis.csv",
                                                                  desired_windows_dimension=5)