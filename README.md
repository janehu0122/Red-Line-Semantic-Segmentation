# Red Line Semantic Segmentation

## Requirements
The program was run using Python 3.8.8. Must have TensorFlow 2.5.0 or greater and Keras 2.3.1 or greater. If packages are not installed, can install through running "pip install tensorflow==2.5.0" and "pip install keras==2.3.1" in main command window

## Files: 
* **Faint Red Line Images** - Testing images containing faint red lines 
* **Red Line Data Files** - Training images along with their corressponding masks 
* **Red Line Images** - All images
* *prediction_generator.py* - Generates and displays predictions for images within a folder
* *unetplusplus_model.py* - Unet ++ architecture
* *main.py* - Code to run to train and test model
  * Saves model as "redline_segmentation_model.hdf5" 

## To Run Program:
1. Download all files
2. Run *main.py* to create and train model
3. Instructions on how to change training and testing set are within main.py



