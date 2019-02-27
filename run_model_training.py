import pandas as pd
from preprocessing.get_input_data import input_data
from keras_model import keras_model
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

epochs = 40
batch_size = 1000
lr = 0.001


def load_model_weights(name, model):
    try:
        model.load_weights(name)
        print('mode loaded')
    except:
        print("Can't load weights!")


def save_model_weights(name, model):
    try:
        model.save_weights(name)
        print("saved classifier weights")
    except:
        print("failed to save classifier weights")
    pass


df = pd.read_csv("HASYv2/hasy-data-labels.csv")

X_train, Y_train, X_test, Y_test, unique_labels = input_data(df, df.shape[0], dataset='HASYv2', test_size=0.2)
n_output = Y_train.shape[1]


model = keras_model(n_output)
# model.summary()

## load weights
# load_model_weights('weights/keras_weights.h5', model)
load_model_weights('weights.best.hdf5', model)
optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)

# compile model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

## no image augmentation
# training = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), callbacks=callbacks_list, verbose=2)

## no image augmentation with learning rate reduction
# training = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), callbacks=[learning_rate_reduction], verbose=2)

## image augmentation
training = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_test, Y_test), steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[learning_rate_reduction], verbose=2)



save_model_weights('weights/keras_weights.h5', model)
model.save('keras.h5')

# Do the followings to convert to a web format
# !pip install tensorflowjs
## !mkdir model # if not exist
#!tensorflowjs_converter --input_format keras keras.h5 model/
