# Red Line Semantic Segmentation

## Requirements
The program was run using Python 3.8.8. Must have TensorFlow 2.5.0 or greater and Keras 2.3.1 or greater. If packages are not installed, can install through running "pip install tensorflow==2.5.0" and "pip install keras==2.3.1" in main command window

## Files: 
* **Faint Red Line Images - Synthetic**: Testing set for synthetic data
* **Red Line Data Files**: Prepocessed training sets for synthetic and hand data
  * New Images: Input images for hand data
  * New Labels: Input masks for hand data
  * Skeleton Images: Input images for synthetic data
  * Skeleton Labels: Input labels for synthetic data
* **Red Line Images - Synthetic**: All of the synthetic images
* **Testing Images - Hand**: Testing set for hand data
* **Training Images - Hand**: Unprocessed training set for hand data
* *prediction_generator.py*: Generates and displays predictions for images within a folder
  * See file for instructions on how to output binary and probability masks
* *unetplusplus_model.py*: Unet ++ architecture
* *main_hand.py*: Run to train model with synthetic data
* *main_synthetic.py*: Run to train model with hand data
  * Saves model as "redline_segmentation_model.hdf5" 
* *environment.yml* - Information regarding environment

## To Run Program:
1. Download all files
2. Run
   * *main_hand.py*
   * *main_synthetic.py*
   
## To Do
* Create a user interface that enables a user to label images, use the images to train a model, and the test the resulting model.

