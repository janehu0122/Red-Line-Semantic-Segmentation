import os
import PIL
from PIL import Image
from PIL import ImageEnhance
import random
from torchvision import transforms

input_dir = "Red Line Data Files/Skeleton Images"
target_dir = "Red Line Data Files/Skeleton Labels"

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir)
        if fname.endswith(".PNG")])

target_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir)
        if fname.endswith(".PNG") and not fname.startswith(".")])

"""
Generating Flipped Images
"""

for j, path in enumerate(input_img_paths):
    
    im = Image.open(input_img_paths[j])
    im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    name = "flipped_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Flipped Images", name))
    
for j, path in enumerate(target_img_paths):
    
    im = Image.open(target_img_paths[j])
    im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    name = "flipped_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Flipped Labels", name))
    
"""
Generating Flipped Images
"""
for j, path in enumerate(input_img_paths):
    
    im = Image.open(input_img_paths[j])
    im = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    name = "flipped_img2_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Flipped Images 2", name))
    
for j, path in enumerate(target_img_paths):
    
    im = Image.open(target_img_paths[j])
    im = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    name = "flipped_img2_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Flipped Labels 2", name))
    
"""
Generating Colored Images
"""

def color(img):
    
    con_enhancer = ImageEnhance.Contrast(img)
    img = con_enhancer.enhance(random.uniform(0.5, 1.5))

    sat_enhancer = ImageEnhance.Color(img)
    img = sat_enhancer.enhance(random.uniform(0.5, 1.5))

    sharp_enhancer = ImageEnhance.Sharpness(img)
    img = sharp_enhancer.enhance(random.uniform(0.5, 1.5))

    bright_enhancer = ImageEnhance.Brightness(img)
    img = bright_enhancer.enhance(random.uniform(0.5, 1.5))

    return img

for j, path in enumerate(input_img_paths):
    
    im = Image.open(input_img_paths[j])
    im = color(im)

    name = "colored_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Colored Images", name))
    
for j, path in enumerate(target_img_paths):
    
    im = Image.open(target_img_paths[j])
    name = "colored_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Colored Labels", name))

"""
Generating Brightened Images
"""

def color(img):
    
    bright_enhancer = ImageEnhance.Brightness(img)
    img = bright_enhancer.enhance(random.uniform(1, 1.5))

    return img

for j, path in enumerate(input_img_paths):
    
    im = Image.open(input_img_paths[j])
    im = color(im)

    name = "colored_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Brightened Images", name))
    
for j, path in enumerate(target_img_paths):
    
    im = Image.open(target_img_paths[j])
    name = "colored_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Brightened Labels", name))
    
"""
Generating Hue Images
"""
def transform(im):

    loader_transform4 = transforms.ColorJitter(hue = 0.5)
    im = loader_transform4(im)

    return im

for j, path in enumerate(input_img_paths):
    
    im = Image.open(input_img_paths[j])
    im = transform(im)

    name = "hue_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Hue Images", name))
    
for j, path in enumerate(target_img_paths):
    
    im = Image.open(target_img_paths[j])
    name = "hue_img_" +  str(j) + ".png"
    im.save(os.path.join("Augmented Images/Hue Labels", name))