# A Simple Real Time Facial Expression Classifier

### Purpose
Recognizing three basic (neutral, happy and sad ) facial expressions by using SVM classifier.
### Dataset
The Cohn-Kanade Facial Expression Database is a facial expressions and action unit database. Image data consist of approximately 500 image sequences from 100 subjects.  For this work, some examples of 3 basic emotion (netural, happy and sad) was used. Last column of the created dataset (64x64.xlsx) shows the classes: 0: Neutral, 1: Sad, 2: Happy 

(see: https://www.ri.cmu.edu/project/cohn-kanade-au-coded-facial-expression-database/)


### Method for classification the expressions
Support vector machine classifier was used for the classify the images. 
### Method for detecting a face in the image
Haar cascade classifier (Viola-Jones algorithm) was used for the face detection in the OpenCV library. 

(see : https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
### Features 
There is no any feature extraction step. Images was used by flattened:
   - Read the image
   - Resize to(64x64)
   - Flatten the image
  
### Dimentional Reduction
Principal Component Analysis - PCA was used

### Tools and Libraries
 - Image Processing : OpenCV
 - Machine Learning: Scikit-Learn
 - Visualization: Seaborn
