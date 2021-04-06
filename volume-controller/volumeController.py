from handDetector import HandDetector
import cv2
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

handDetector = HandDetector(min_detection_confidence=0.7)
webcamFeed = cv2.VideoCapture(0)

#Volume related initializations
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#print(volume.GetVolumeRange()) --> (-65.25, 0.0)

while True:
    status, image = webcamFeed.read()
    handLandmarks = handDetector.findHandLandMarks(image=image, draw=True)

    if(len(handLandmarks) != 0):
        #for volume control we need 4th and 8th landmark
        #details: https://google.github.io/mediapipe/solutions/hands
        x1, y1 = handLandmarks[4][1], handLandmarks[4][2]
        x2, y2 = handLandmarks[8][1], handLandmarks[8][2]
        length = math.hypot(x2-x1, y2-y1)
        print(length)

        #Hand range(length): 50-250
        #Volume Range: (-65.25, 0.0)

        volumeValue = np.interp(length, [50, 250], [-65.25, 0.0]) #coverting length to proportionate to volume range
        volume.SetMasterVolumeLevel(volumeValue, None)


        cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cv2.imshow("Volume", image)
    cv2.waitKey(1)