import cv2

webcamFeed = cv2.VideoCapture(0)

tracker = cv2.TrackerGOTURN_create()     #initialize Tracker
status, image = webcamFeed.read()       #display a image to user to select the region
roi = cv2.selectROI("Select Object to track", image, False)   #Get ROI that user selected
tracker.init(image, roi)            #Add ROI to to tracker

while True:
    status, image = webcamFeed.read()
    success, bbox = tracker.update(image)

    if success:
        #drawing bounding box on image
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(image, (x, y), ((x+w), (y+h)), (255, 0, 255), 3, 1)
        cv2.putText(image, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
    else:
        cv2.putText(image, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

    cv2.imshow("Output", image)
    cv2.waitKey(1)