"""
Title: Prediction Generator for Red Line Segmentation
Description: Helper function to generate all predicition masks on a folder
"""
import os
import PIL
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
import numpy as np


def generate_predictions_on_folder(folder_path, unet, img_size):
    """ Function to display model predicitons on images within a folder"""
    
    testing_dir = folder_path

    testing_img_paths = [os.path.join(testing_dir, fname) 
                         for fname in os.listdir(testing_dir)
                         if fname.endswith(".png")]

    x = np.zeros((len(testing_img_paths),) + img_size + (3,), dtype="float32")

    for j, path in enumerate(testing_img_paths):
                img = load_img(path)
                # cropping images from 900x720 to 512x512
                img = img.crop(box=(313,99,825,611))
                # resizing image from 512x512 to 256x256
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
        """ Utility to display the original image. """
        image = PIL.Image.open(testing_img_paths[i])
        image = image.crop(box=(313,99,825,611))
        image = image.resize((256,256))
        display(image)

    # displaying all predictions for images in a folder
    for i in range(0,len(testing_img_paths)):
        # Display input image
        display_cropped_img(i)
        # Display mask predicted by our model
        display_mask(i)