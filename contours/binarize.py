import cv2
import numpy as np


def edgeDetection(image, sigma=0.33):
    #lower sigma-->tighter threshold
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))
    edged = cv2.Canny(image, lower, upper)
    return edged


def thresholding(image):
    #internally used by edge detection algorithm
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (7, 7), 3)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, image = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image


def segmentation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(hsv, np.array([-10, 50, 100]), np.array([50, 150, 225]))
    return image