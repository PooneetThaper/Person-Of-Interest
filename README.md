# WheresObama

## Description
This program uses transfer learning on Google's Inception v3 image classification model and OpenCV to detect Former President Obama in images. It does so by using OpenCV to find all faces in the images and the retrained image classification model to decide whether or not that face is President Obama. 

## Usage
$ python3 face_detect_cv3.py [label] [path/to/image]
(for available labels see ClassificationModel folder)

## Dependencies
* TensorFlow
* OpenCV3
* Python3

## Credits
* @shantnu for the facial detection code 
