import cv2
import numpy as np

#### Loading the yolo algorithm ####
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# Saving names of object classes in list object_classes from the file coco.names
object_classes = []
with open("coco.names", "r") as f:
    object_classes = [line.strip() for line in f.readlines()]
# Getting layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in yolo.getUnconnectedOutLayers()]

#### Loading Image ####
# Reading the image
my_image = cv2.imread("room.jpg")
# Resizing the image
my_image = cv2.resize(my_image, None, fx=0.5, fy=0.5)
height, width, channels = my_image.shape

#### Detecting objects ####
# Creating a blob
blob = cv2.dnn.blobFromImage(image=my_image,
                             scalefactor=1/255,
                             size=(1664, 1664),
                             mean=(0, 0, 0),
                             swapRB=True,
                             crop=False)
yolo.setInput(blob)
outputs = yolo.forward(output_layers)

#### Displaying the information ####
boxes = []
confidences = []
class_ids = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object has been detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # creating the rectangles
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
# Applying non-max supression to remove repeated boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
#### Displaying the label detected objects ####
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(object_classes[class_ids[i]])+ "  " + str(confidences[i])
        cv2.rectangle(my_image, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(my_image, label, (x,y+30), font, 1, (0,0,0), 2)
        print(label)


cv2.imshow("Image", my_image)
cv2.waitKey(0)
cv2.destroyAllWindows()