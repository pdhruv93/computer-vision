import cv2
from faceDetectorTrackerUtil import FaceDetectorTrackor
import time


#Declarations
WEIGHT_FILE = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
MIN_CONFIDENCE = 0.5

webcamFeed = cv2.VideoCapture(0)
dt = FaceDetectorTrackor(WEIGHT_FILE, MODEL_FILE, MIN_CONFIDENCE)


while True:
    status, image = webcamFeed.read()
    #returns coordinate of bounding box
    #detector.detect(image, drawCenter=True, drawBox=True)

    #returns a list of (coordinate of center) and (area of bouding box) till current frame
    dt.track(image, draw=True)

    #Enable below code if you want to use face tracking application for robots movement
    #To use tracking as well as findDirection module at the same time, create separate threads for each
    #direction = dt.findDirection()
    #print(direction)
    #print("========================================================")
    #time.sleep(5)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(1)
