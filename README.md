# Person-Of-Interest

## Description
This program uses transfer learning on Google's Inception v3 image classification model and OpenCV to detect the specified person in the specified image. It does so by using OpenCV to find all faces in the images and using the retrained image classification model to decide whether or not that face is of the specified person. 

## Usage
$ python3 face_detect_cv3.py [label] [path/to/image]

for available labels see ClassificationModel folder
Ex: python3 face_detect_cv3.py obama [path/to/image]

## Dependencies
* TensorFlow
* OpenCV3
* Python3

## Credits
* @shantnu for the facial detection tutorial
