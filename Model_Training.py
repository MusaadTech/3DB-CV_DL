import os
from glob import glob

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = INFO, WARNING, and ERROR

from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

NUMBER_CLASSES = 10
activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}


def get_cv2_image(path, img_rows, img_cols, color_type=3):
    """
    Function that return an opencv image from the path and the right number of dimension
    """
    if color_type == 1:  # Loading as Grayscale image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3:  # Loading as color image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_rows, img_cols))  # Reduce size
    return img


# Loading Training dataset
def load_train(img_rows, img_cols, color_type=3):
    """
    Return train images and train labels from the original path
    """
    train_images = []
    train_labels = []
    # Loop over the training folder
    for classed in tqdm(range(NUMBER_CLASSES)):
        print('Loading directory c{}'.format(classed))
        files = glob(os.path.join('D:/Kaggle/imgs/train/c' + str(classed), '*.jpg'))
        for file in files:
            img = get_cv2_image(file, img_rows, img_cols, color_type)
            train_images.append(img)
            train_labels.append(classed)
    return train_images, train_labels


def read_and_normalize_train_data(img_rows, img_cols, color_type):
    """
    Load + categorical + split
    """
    X, labels = load_train(img_rows, img_cols, color_type)
    y = np_utils.to_categorical(labels, 10)  # categorical train label
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # split into train and test
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)

    return x_train, x_test, y_train, y_test


# Loading validation dataset
def load_test(size=200000, img_rows=64, img_cols=64, color_type=3):
    """
    Same as above but for validation dataset
    """
    path = os.path.join('D:/Kaggle/imgs/test', '*.jpg')
    files = sorted(glob(path))
    X_test, X_test_id = [], []
    total = 0
    files_size = len(files)
    for file in tqdm(files):
        if total >= size or total >= files_size:
            break
        file_base = os.path.basename(file)
        img = get_cv2_image(file, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(file_base)
        total += 1
    return X_test, X_test_id


def read_and_normalize_sampled_test_data(size, img_rows, img_cols, color_type=3):
    test_data, test_ids = load_test(size, img_rows, img_cols, color_type)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(-1, img_rows, img_cols, color_type)
    return test_data, test_ids


img_rows = 64  # dimension of images
img_cols = 64
color_type = 1  # grey
nb_test_samples = 200

# loading train images
x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols, color_type)

# loading validation images
test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)

# Number of batch size and epochs
batch_size = 40
nb_epoch = 6

models_dir = "saved_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Proposed selected model.
def create_model():
    model = Sequential()

    ## CNN 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, color_type)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    ## CNN 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    ## CNN 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    ## Output
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = create_model()

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=nb_epoch, batch_size=batch_size, verbose=1)


# Second proposed model
def create_model_2():
    model_2 = Sequential()

    ## CNN 1
    model_2.add(Conv2D(32, 3, 3, padding='same', input_shape=(img_rows, img_cols, color_type)))
    model_2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_2.add(Dropout(0.5))

    ## CNN 2
    model_2.add(Conv2D(64, 3, 3, padding='same'))
    model_2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_2.add(Dropout(0.5))

    ## CNN 3
    model_2.add(Conv2D(128, 3, 3, padding='same'))
    model_2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_2.add(Dropout(0.5))

    ## Output
    model_2.add(Flatten())
    model_2.add(Dense(10, activation='softmax'))

    model_2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_2


model_2 = create_model_2()

history_2 = model_2.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=nb_epoch, batch_size=batch_size, verbose=1)


# Third proposed model
def create_model_3():
    model_3 = Sequential()

    ## CNN 1
    model_3.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_rows, img_cols, color_type)))
    model_3.add(BatchNormalization())
    model_3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_3.add(BatchNormalization(axis=3))
    model_3.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_3.add(Dropout(0.3))

    ## CNN 2
    model_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_3.add(BatchNormalization())
    model_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_3.add(BatchNormalization(axis=3))
    model_3.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_3.add(Dropout(0.5))

    ## Output
    model_3.add(Flatten())
    model_3.add(Dense(512, activation='relu'))
    model_3.add(BatchNormalization())
    model_3.add(Dropout(0.5))
    model_3.add(Dense(128, activation='relu'))
    model_3.add(Dropout(0.25))
    model_3.add(Dense(10, activation='softmax'))

    model_3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_3


model_3 = create_model_3()

history_3 = model_3.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=nb_epoch, batch_size=batch_size, verbose=1)

score1 = model.evaluate(x_test, y_test, verbose=1)

score2 = model_2.evaluate(x_test, y_test, verbose=1)

score3 = model_3.evaluate(x_test, y_test, verbose=1)

model.save("saved_models/selected_model/Model_1")

model_2.save("saved_models/compared_model/Model_2")

model_3.save("saved_models/compared_model/Model_3")
