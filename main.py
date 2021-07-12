"""
Title: Red Line segmentation with a U-Net++ like architecture
Description: Image segmentation model trained on the red line dataset.
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
import prediction_generator
import unetplusplus_model

### Binary Skeleton training set: "Skeleton Images", "Skeleton Labels"
### Original training set: "Images", "Labels"

input_dir = "Red Line Data Files/New Images 3"
target_dir = "Red Line Data Files/New Labels 3"
img_size = (256, 256)
num_classes = 2
batch_size = 10
epochs = 150
number_of_filters = 2

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if (fname.endswith(".PNG") or fname.endswith(".png"))
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if (fname.endswith(".PNG") or fname.endswith(".png")) and not fname.startswith(".")
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
            # y[j] = np.where(y[j]<255, 0, 1)
            #ground truth labels are from 0-255, need to covert labels to 0,
            y[j] = np.where(y[j]>0, 1, 0)
        return x, y

"""
## Prepare U-Net Xception-style model
"""    
unet = unetplusplus_model.UNetPP(number_of_filters, num_classes, batch_size, epochs)
unet.model.summary()

"""
## Set aside a validation split
"""

# Split our img paths into a training and a validation set
val_samples = 10
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
## Save the model 
"""
unet.model.save("redline_segmentation_model.hdf5")

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
    
# Display results for validation image #3
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

# folder containing images to run through model
# for testing on faint red lines, "Faint Red Line Images"
# for testing on all images, "Red Line Images"
testing_dir = "Testing Images 2"

testing_folder_paths = [os.path.join(testing_dir, fname)
        for fname in os.listdir(testing_dir)
        if fname.endswith("mm")]

for folder_path in testing_folder_paths:
    prediction_generator.generate_predictions_on_folder(folder_path, unet, img_size)
            
