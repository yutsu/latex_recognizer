import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def prepareImages(data, m, dataset='HASYv2'):
    """
    dataset: hasy-data
    """
    print("Preparing images")
    X_data = np.zeros((m, 32, 32, 3))
    count = 0

    for fig in data['path']:
        # load images into images of size 32x32x1
        # v2-00000 to v2-168232
        img = image.load_img(dataset + "/" + fig, target_size=(32, 32, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_data[count] = x
        if (count % 5000 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    # for i in range(168233):
        # load images into images of size 32x32x1
        # v2-00000 to v2-168232

        # fig = "v2-" + str(i).zfill(5)
        # img = image.load_img("../HASYv2/" + dataset + "/" + fig, target_size=(32, 32, 1))
        # x = image.img_to_array(img)
        # x = preprocess_input(x)

        # X_train[i] = x
        # if (i % 5000 == 0):
        #     print("Processing image: ", i + 1, ", ", fig)

    return X_data


def input_data(data, m, dataset='HASYv2', test_size=0.2):
    """
    data: df = pd.read_csv("HASYv2/hasy-data-labels.csv")
    m: df.shape[0]
    """

    x_data = prepareImages(data, m, dataset=dataset)
    x_data /= 255

    y = data['symbol_id']
    unique_labels = list(set(y))
    # len(unique_labels)  #369
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    new_labels = np.array([labels_dict[l] for l in y], dtype=np.int32)

    # latex corresponding to smbol_id

    x_shuffle, new_labels_shuffle = shuffle(x_data, new_labels, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x_shuffle, new_labels_shuffle, test_size=test_size, random_state=0)

    # reshape to (32, 32, 3)
    X_train = x_train.reshape(
        (x_train.shape[0], x_train.shape[1], x_train.shape[2], 3))
    X_test = x_test.reshape(
        (x_test.shape[0], x_test.shape[1], x_test.shape[2], 3))

    # onehot encoding
    n_classes = len(unique_labels)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)

    return X_train, Y_train, X_test, Y_test, unique_labels
