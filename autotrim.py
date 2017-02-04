import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
import cv2
import os

'''
    This script was written to go through a directory of images and crop each
    to only the face since the image classifier is only provided with the found
    faces so this should help with accuracy

    put photos to trim faces from in trim/ folder
    photos that are not trimmed (ie photos that dont satisfy # of required faces found) are stored in trimFail/
    successfully trimmed photos are stored in trimmed/
'''


cascPath = 'cascade/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

successes = 0
fails = 0

# Go though all images in file trim
for file in os.listdir('trim'):
    print(file)

    # Load image and convert to grayscale
    image = cv2.imread(os.path.join('trim/',file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.2,
      minNeighbors=5,
      minSize=(30, 30)
      #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))

    # If it found more than one face or no faces, write file to fail directory
    if len(faces) != 1: #This was used for photos where only one face was expected (ie the face that we need to classify)
    #if len(faces) == 0:
      cv2.imwrite(os.path.join('trimFail/',file),image)
      fails += 1
      continue
    # Else write to successfully trimmed folder
    else:
        successes += 1
        for (x, y, w, h) in faces:
            yPadding = h/5
            xPadding = w/5

            leftLim = x-xPadding
            if leftLim < 0:
                leftLim = 0

            rightLim = x+w+xPadding
            if rightLim > image.shape[1]:
                rightLim = image.shape[1] - 1

            topLim = y-yPadding
            if topLim < 0:
                topLim = 0

            botLim = y+h+yPadding
            if botLim > image.shape[0]:
                botLim = image.shape[0] - 1

            cv2.imwrite(os.path.join('trimmed/',file),image[topLim:(botLim),leftLim:(rightLim)])

print ("Successes:{0}".format(successes))
print ("Fails:{0}".format(fails))
