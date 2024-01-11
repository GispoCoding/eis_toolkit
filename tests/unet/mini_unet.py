import numpy as np

from eis_toolkit.unet.mini_unet import train_and_predict_the_model


def test_the_mini_unet_with_uncertainty():
    """This is the test with uncertainty."""
    random_images_training = np.random.randint(0, 100, size=(1000, 32, 32, 3))
    labels_training = np.random.randint(0, 1, size=(1000, 1, 1, 3))
    random_images_testing = np.random.randint(0, 100, size=(1000, 32, 32, 3))
    labels_testing = np.random.randint(0, 1, size=(1000, 1, 1, 3))

    prediction = train_and_predict_the_model(
        x_train=random_images_training,
        x_test=random_images_testing,
        y_train=labels_training,
        y_test=labels_testing,
        epochs=2,
        batch_size=32,
        is_uncertainty=True,
        list_of_convolutional_layers=[32, 64, 128],
        dropout=0.2,
        pool_size=2,
        up_sampling_factor=2,
        output_filters=2,
        last_activation="sigmoid",
        output_kernel=(1, 1),
        data_augmentation=False,
        regularization=None,
        uncertainty_coefficient=0.2,
    )
    assert prediction.shape[0] != 0


def test_the_mini_unet_with_no_uncertainty():
    """This is the test without uncertainty."""
    random_images_training = np.random.randint(0, 100, size=(1000, 32, 32, 3))
    labels_training = np.random.randint(0, 1, size=(1000, 1, 1, 2))
    random_images_testing = np.random.randint(0, 100, size=(1000, 32, 32, 3))
    labels_testing = np.random.randint(0, 1, size=(1000, 1, 1, 2))

    prediction = train_and_predict_the_model(
        x_train=random_images_training,
        x_test=random_images_testing,
        y_train=labels_training,
        y_test=labels_testing,
        epochs=2,
        batch_size=32,
        is_uncertainty=False,
        list_of_convolutional_layers=[32, 64, 128],
        dropout=0.2,
        pool_size=2,
        up_sampling_factor=2,
        output_filters=2,
        last_activation="sigmoid",
        output_kernel=(1, 1),
        data_augmentation=False,
        regularization=None,
        uncertainty_coefficient=0.2,
    )
    assert prediction.shape[0] != 0
