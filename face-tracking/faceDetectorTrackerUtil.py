import cv2
import numpy as np
from collections import deque

class FaceDetectorTrackor:
    def __init__(self, weight_file, model_file, min_confidence=0.5, tracking_points=40):
        # Loading dnn model files from open-cv
        self.model = cv2.dnn.readNetFromCaffe(weight_file, model_file)
        self.min_confidence = min_confidence
        self.trackingDetails = deque(maxlen=tracking_points)

    def detect(self, image, drawBox=False, drawCenter=False):
        h, w = image.shape[:2]
        # Preparing BLOB from Image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        # Reference to blob fn:https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
        # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
        # scalefactor: how much the intensisties must be decreased for the image(*1/sf)
        # size: targetSize, usually the size that the model accepts
        # mean: mean subtraction values, he supplied value is subtracted from every channel of the image, swabRB must be True
        # swapRB: openCV uses BGR channel order; however `mean` value uses RGB order

        # The main part: Detecting faces in image
        self.model.setInput(blob)
        detections = self.model.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence > self.min_confidence:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if drawBox:
                    # draw the bounding box of the face along with the associated probability
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                if drawCenter:
                    #draw the center point of box
                    cv2.circle(image, (int((startX + endX) / 2), int((startY + endY) / 2)), 15, (255, 0, 255), cv2.FILLED)

        return [startX, startY, endX, endY]


    def track(self, image, draw=False):
        try:
            bbox = self.detect(image)
            # print(bbox)

            if len(bbox) != 0:
                centerX, centerY = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                self.trackingDetails.append([(centerX, centerY), area])
                # print(centerX, centerY)

            if draw:
                # Tracking lines
                for i in range(1, len(self.trackingDetails)):
                    cv2.line(image, self.trackingDetails[i - 1][0], self.trackingDetails[i][0], (0, 0, 255), 3)
        except:
            pass


    def findDirection(self):
        direction = "same"
        length = len(self.trackingDetails)
        try:
            lastX, lastY = self.trackingDetails[length - 2][0]
            currentX, currentY  = self.trackingDetails[length-1][0]
            areaLast = self.trackingDetails[length - 2][1]
            areaNow = self.trackingDetails[length-1][1]

            print(lastX, lastY, areaLast)
            print(currentX, currentY, areaNow)

            if(currentX < lastX - 100):
                direction = "left"  # person has moved to left as compared to last position-->move robot left
            elif (currentX > lastX + 100):
                direction = "right"  # person has moved to right as compared to last position-->move robot right

            if(areaNow > areaLast + 3000):
                direction = direction + "_backward"    #area has increased-->face has come closer to camera-->move robot backwards
            elif(areaNow < areaLast - 3000):
                direction = direction + "_forward"    #area has decreased-->face has moved away from camera-->move robot forward

        except:
            pass

        return direction