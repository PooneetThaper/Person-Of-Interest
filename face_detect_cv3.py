import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2

# Get user supplied values
cascPath = 'cascade/haarcascade_frontalface_default.xml'
label= sys.argv[1]
imagePath = sys.argv[2]


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Take all found faces and put them in p for later saving
p = []
for (x, y, w, h) in faces:
    # padding used to get a wider angle view of face for classification
    yPadding = int(h/4.5)
    xPadding = int(w/4.5)

    # padding might go out of bounds so this is here to prevent that
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
    p.append(image[topLim:botLim,leftLim:rightLim])

# Import tensorflow for image recognition
import tensorflow as tf, sys

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("ClassificationModel/{}/retrained_labels.txt".format(label))]

# Unpersists graph from file
with tf.gfile.FastGFile("ClassificationModel/{}/retrained_graph.pb".format(label), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Save all images and reload into tensorflow
image_data = []
for i in range(len(p)):
    write_path = "temp/{0}.jpg".format(i)
    cv2.imwrite(write_path,p[i])
    image_data.append(tf.gfile.FastGFile(write_path, 'rb').read())

# Keeps track of all
successes = []
num = 0

with tf.Session() as sess:
    for i in range(len(p)):
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data[i]})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]


        # If found save as a success and increase num (else save as fail)
        # Positive result requires at least 80% confidence
        if label_lines[top_k[0]] == label and predictions[0][top_k[0]] > 0.75:
            successes.append(1)
            num = num + 1;
        else:
            successes.append(0)

        print("Checking face #{}: {}".format(i+1,label_lines[top_k[0]]))
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))




print("Found {} {}(s)!".format(num,label))

#prepareing to resize image for large images
r = 500.0 / image.shape[0]
dim = (int(image.shape[1] * r),500)

i=0
for (x, y, w, h) in faces:
    if successes[i]==1:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), int(2/r))
    else:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), int(2/r))
    i = i + 1;


# Resizeing image for display on screen (in case of huge image)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Faces found (Green = Obama, Red = Not Obama)", image)
cv2.waitKey(0)
