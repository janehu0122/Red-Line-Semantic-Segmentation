# Red Line Semantic Segmentation

## Requirements
The program was run using Python 3.8.8. Must have TensorFlow 2.5.0 or greater and Keras 2.3.1 or greater. If packages are not installed, can install through running "pip install tensorflow==2.5.0" and "pip install keras==2.3.1" in main command window

## Files: 
* **Faint Red Line Images - Synthetic** 
* **Red Line Data Files**
  * New Images
  * New Labels
  * Skeleton Images
  * Skeleton Labels
* **Red Line Images - Synthetic**
* **Testing Images - Hand**
* **Training Images - Hand** 
* *prediction_generator.py* - Generates and displays predictions for images within a folder
* *unetplusplus_model.py* - Unet ++ architecture
* *main_hand.py* 
* *main_synthetic.py* 
  * Saves model as "redline_segmentation_model.hdf5" 
* *environment.yml* - Information regarding environment

## To Run Program:
1. Download all files
2. Run
   * *main_hand.py*
   * *main_synthetic.py*
   
## To Do
* Create a user interface that enables a user to label images, use the images to train a model, and the test the resulting model.

