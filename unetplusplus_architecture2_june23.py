"""
Title: Image segmentation with a U-Net++ like architecture Attempt 2
Description: Image segmentation model trained from scratch on the Red Line dataset.

Sources: 
"""

"""
## Prepare paths of input images and target segmentation masks
"""

import os
import PIL
from PIL import ImageOps
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import random

input_dir = "Red Line Data Files/Images"
target_dir = "Red Line Data Files/Labels"
img_size = (256, 256)
num_classes = 2
batch_size = 5

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".PNG")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".PNG") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

"""
## Prepare `Sequence` class to load & vectorize batches of data
"""

class RedLines(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] = np.where(y[j]<255, 0, 1)
        return x, y

"""
## Prepare U-Net Xception-style model
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import typing


def weighted_loss(original_loss_function: typing.Callable, weights_list: dict) -> typing.Callable:
    """
    Help function to balance background, bacteria and blood cells.
    """
    def loss_function(true, pred):
        class_selectors = tf.cast(K.argmax(true, axis=-1), tf.int32)
        class_selectors = [K.equal(i, class_selectors) for i in range(len(weights_list))]
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_function(true, pred)
        loss = loss * weight_multiplier
        return loss
    return loss_function


@tf.function
def loss(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1, dice_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) \
           + iou_weight * log_iou(y_true, y_pred, smooth) \
           + dice_weight * log_dice(y_true, y_pred, smooth)

@tf.function
def log_iou(y_true, y_pred, smooth=1):
    return - K.log(iou(y_true, y_pred, smooth))


@tf.function
def log_dice(y_true, y_pred, smooth=1):
    return -K.log(dice(y_true, y_pred, smooth))


@tf.function
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


@tf.function
def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam

# Some hyper parameters
epochs = 75
number_of_filters = 2

def conv2d(filters: int):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  padding='same')


def conv2dtranspose(filters: int):
    return Conv2DTranspose(filters=filters,
                           kernel_size=(2, 2),
                           strides=(2, 2),
                           padding='same')


class UNetPP:
    def __init__(self):
        model_input = Input((256, 256, 3))
        x00 = conv2d(filters=int(16 * number_of_filters))(model_input)
        x00 = BatchNormalization()(x00)
        x00 = LeakyReLU(0.01)(x00)
        x00 = Dropout(0.2)(x00)
        x00 = conv2d(filters=int(16 * number_of_filters))(x00)
        x00 = BatchNormalization()(x00)
        x00 = LeakyReLU(0.01)(x00)
        # x00 = Dropout(0.2)(x00)
        p0 = MaxPooling2D(pool_size=(2, 2))(x00)

        x10 = conv2d(filters=int(32 * number_of_filters))(p0)
        x10 = BatchNormalization()(x10)
        x10 = LeakyReLU(0.01)(x10)
        x10 = Dropout(0.2)(x10)
        x10 = conv2d(filters=int(32 * number_of_filters))(x10)
        x10 = BatchNormalization()(x10)
        x10 = LeakyReLU(0.01)(x10)
        # x10 = Dropout(0.2)(x10)
        p1 = MaxPooling2D(pool_size=(2, 2))(x10)

        x01 = conv2dtranspose(int(16 * number_of_filters))(x10)
        x01 = concatenate([x00, x01])
        x01 = conv2d(filters=int(16 * number_of_filters))(x01)
        x01 = BatchNormalization()(x01)
        x01 = LeakyReLU(0.01)(x01)
        x01 = conv2d(filters=int(16 * number_of_filters))(x01)
        x01 = BatchNormalization()(x01)
        x01 = LeakyReLU(0.01)(x01)
        # x01 = Dropout(0.2)(x01)

        x20 = conv2d(filters=int(64 * number_of_filters))(p1)
        x20 = BatchNormalization()(x20)
        x20 = LeakyReLU(0.01)(x20)
        x20 = Dropout(0.2)(x20)
        x20 = conv2d(filters=int(64 * number_of_filters))(x20)
        x20 = BatchNormalization()(x20)
        x20 = LeakyReLU(0.01)(x20)
        # x20 = Dropout(0.2)(x20)
        p2 = MaxPooling2D(pool_size=(2, 2))(x20)

        x11 = conv2dtranspose(int(16 * number_of_filters))(x20)
        x11 = concatenate([x10, x11])
        x11 = conv2d(filters=int(16 * number_of_filters))(x11)
        x11 = BatchNormalization()(x11)
        x11 = LeakyReLU(0.01)(x11)
        x11 = conv2d(filters=int(16 * number_of_filters))(x11)
        x11 = BatchNormalization()(x11)
        x11 = LeakyReLU(0.01)(x11)
        # x11 = Dropout(0.2)(x11)

        x02 = conv2dtranspose(int(16 * number_of_filters))(x11)
        x02 = concatenate([x00, x01, x02])
        x02 = conv2d(filters=int(16 * number_of_filters))(x02)
        x02 = BatchNormalization()(x02)
        x02 = LeakyReLU(0.01)(x02)
        x02 = conv2d(filters=int(16 * number_of_filters))(x02)
        x02 = BatchNormalization()(x02)
        x02 = LeakyReLU(0.01)(x02)
        # x02 = Dropout(0.2)(x02)

        x30 = conv2d(filters=int(128 * number_of_filters))(p2)
        x30 = BatchNormalization()(x30)
        x30 = LeakyReLU(0.01)(x30)
        x30 = Dropout(0.2)(x30)
        x30 = conv2d(filters=int(128 * number_of_filters))(x30)
        x30 = BatchNormalization()(x30)
        x30 = LeakyReLU(0.01)(x30)
        # x30 = Dropout(0.2)(x30)
        p3 = MaxPooling2D(pool_size=(2, 2))(x30)

        x21 = conv2dtranspose(int(16 * number_of_filters))(x30)
        x21 = concatenate([x20, x21])
        x21 = conv2d(filters=int(16 * number_of_filters))(x21)
        x21 = BatchNormalization()(x21)
        x21 = LeakyReLU(0.01)(x21)
        x21 = conv2d(filters=int(16 * number_of_filters))(x21)
        x21 = BatchNormalization()(x21)
        x21 = LeakyReLU(0.01)(x21)
        # x21 = Dropout(0.2)(x21)

        x12 = conv2dtranspose(int(16 * number_of_filters))(x21)
        x12 = concatenate([x10, x11, x12])
        x12 = conv2d(filters=int(16 * number_of_filters))(x12)
        x12 = BatchNormalization()(x12)
        x12 = LeakyReLU(0.01)(x12)
        x12 = conv2d(filters=int(16 * number_of_filters))(x12)
        x12 = BatchNormalization()(x12)
        x12 = LeakyReLU(0.01)(x12)
        # x12 = Dropout(0.2)(x12)

        x03 = conv2dtranspose(int(16 * number_of_filters))(x12)
        x03 = concatenate([x00, x01, x02, x03])
        x03 = conv2d(filters=int(16 * number_of_filters))(x03)
        x03 = BatchNormalization()(x03)
        x03 = LeakyReLU(0.01)(x03)
        x03 = conv2d(filters=int(16 * number_of_filters))(x03)
        x03 = BatchNormalization()(x03)
        x03 = LeakyReLU(0.01)(x03)
        # x03 = Dropout(0.2)(x03)

        m = conv2d(filters=int(256 * number_of_filters))(p3)
        m = BatchNormalization()(m)
        m = LeakyReLU(0.01)(m)
        m = conv2d(filters=int(256 * number_of_filters))(m)
        m = BatchNormalization()(m)
        m = LeakyReLU(0.01)(m)
        # m = Dropout(0.2)(m)

        x31 = conv2dtranspose(int(128 * number_of_filters))(m)
        x31 = concatenate([x31, x30])
        x31 = conv2d(filters=int(128 * number_of_filters))(x31)
        x31 = BatchNormalization()(x31)
        x31 = LeakyReLU(0.01)(x31)
        x31 = conv2d(filters=int(128 * number_of_filters))(x31)
        x31 = BatchNormalization()(x31)
        x31 = LeakyReLU(0.01)(x31)
        # x31 = Dropout(0.2)(x31)

        x22 = conv2dtranspose(int(64 * number_of_filters))(x31)
        x22 = concatenate([x22, x20, x21])
        x22 = conv2d(filters=int(64 * number_of_filters))(x22)
        x22 = BatchNormalization()(x22)
        x22 = LeakyReLU(0.01)(x22)
        x22 = conv2d(filters=int(64 * number_of_filters))(x22)
        x22 = BatchNormalization()(x22)
        x22 = LeakyReLU(0.01)(x22)
        # x22 = Dropout(0.2)(x22)

        x13 = conv2dtranspose(int(32 * number_of_filters))(x22)
        x13 = concatenate([x13, x10, x11, x12])
        x13 = conv2d(filters=int(32 * number_of_filters))(x13)
        x13 = BatchNormalization()(x13)
        x13 = LeakyReLU(0.01)(x13)
        x13 = conv2d(filters=int(32 * number_of_filters))(x13)
        x13 = BatchNormalization()(x13)
        x13 = LeakyReLU(0.01)(x13)
        # x13 = Dropout(0.2)(x13)

        x04 = conv2dtranspose(int(16 * number_of_filters))(x13)
        x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
        x04 = conv2d(filters=int(16 * number_of_filters))(x04)
        x04 = BatchNormalization()(x04)
        x04 = LeakyReLU(0.01)(x04)
        x04 = conv2d(filters=int(16 * number_of_filters))(x04)
        x04 = BatchNormalization()(x04)
        x04 = LeakyReLU(0.01)(x04)
        # x04 = Dropout(0.2)(x04)

        output = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax')(x04)

        self.model = tf.keras.Model(inputs=[model_input], outputs=[output])
        self.optimizer = Adam(lr=0.0005)

    def compile(self, loss_function, metrics=[iou, dice]):
        self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=metrics)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, class_weights=None):
        self.compile(loss_function=weighted_loss(loss, class_weights))
        return self.model.fit(x_train, y_train,
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              validation_data=[x_val, y_val],
                              validation_steps=x_val.shape[0] // batch_size,
                              batch_size=batch_size,
                              epochs=epochs,
                              shuffle=True)
    
unet = UNetPP()
unet.model.summary()

"""
## Set aside a validation split
"""

# Split our img paths into a training and a validation set
val_samples = 5
random.Random(30).shuffle(input_img_paths)
random.Random(30).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = RedLines(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = RedLines(batch_size, img_size, val_input_img_paths, val_target_img_paths)

"""
## Train the model
"""
unet.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("red_line_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
unet.model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

"""
## Visualize predictions on validation set
"""
# Generate predictions for all images in the validation set

val_gen = RedLines(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = unet.model.predict(val_gen)

def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)
    
# Display results for validation image #10
i = 3

# Display input image
display(Image(filename=val_input_img_paths[i]))

# Display ground-truth target mask
img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

# Display mask predicted by our model
display_mask(i)
    
"""
## Visualize Predictions on Faint Line Images
"""

def generate_predictions_on_folder(folder_path):
    
    testing_dir = folder_path

    testing_img_paths = [os.path.join(testing_dir, fname) 
                         for fname in os.listdir(testing_dir)
                         if fname.endswith(".png")]

    x = np.zeros((len(testing_img_paths),) + img_size + (3,), dtype="float32")

    for j, path in enumerate(testing_img_paths):
                img = load_img(path)
                
                ## cropping imagese from 900x720 to 512x512
                img = img.crop(box=(313,99,825,611))
                img = img.resize(img_size)
                
                x[j] = img

    testing_preds = unet.model.predict(x)

    def display_mask(i):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(testing_preds[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        display(img)
        
    def display_cropped_img(i):
        image = PIL.Image.open(testing_img_paths[i])
        image = image.crop(box=(313,99,825,611))
        image = image.resize((256,256))
        display(image)


    for i in range(0,len(testing_img_paths)):
        # Display input image
        display_cropped_img(i)
        # Display mask predicted by our model
        display_mask(i)

testing_dir = "Faint Red Line Images"

testing_folder_paths = [os.path.join(testing_dir, fname)
        for fname in os.listdir(testing_dir)
        if fname.endswith("mm")]

for folder_path in testing_folder_paths:
    generate_predictions_on_folder(folder_path)
            