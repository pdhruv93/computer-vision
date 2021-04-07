import cv2

webcamFeed = cv2.VideoCapture(0)

tracker = cv2.TrackerGOTURN()            #initialize Tracker
status1, image = webcamFeed.read()       #display a image to user to select the region
roi = cv2.selectROI("Select Object to track", image, False)   #Get ROI that user selected
print("3")
tracker.init(image, roi)            #Add ROI to to tracker
print("4")
print("2")

while True:
    print("1")
    status2, image = webcamFeed.read()
    cv2.imshow("Output", image)
    cv2.waitKey(1)