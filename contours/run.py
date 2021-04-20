import cv2
from binarize import edgeDetection
import imutils
print(cv2.__version__)


image = cv2.imread("2.jpg")

# findContours() needs binary image, so creating one using Canny edge
binary_image = edgeDetection(image)

# finding contours from binary image: Since OpenCV 3.2, findContours() no longer modifies the source image
# 3rd argument: contour approximation. A contour is simply a shape represented by a collection of points.
# So this argument specifies how much points should be stored so that the contour shape could be drawn
# CHAIN_APPROX_NONE signifies that we store all the points(no approximation). It takes more storage space
# but if there is a zig zag line, we definitely need to store all the points
# but in case of a straight line, 2 points would be enough as the points in between can be approximated based
# on equation of a straight line, so we use cv.CHAIN_APPROX_SIMPLE

# 2nd argument: contour retrieval mode- https://docs.opencv.org/4.5.1/d9/d8b/tutorial_py_contours_hierarchy.html
# A shape may be present inside another shape. So we may call the outer shape as the parent and
# inner shape as child. This argument decides if want the output in this hierarchical(parent child) manner
# or we simply want them as a list(having no hierarchy)
# Note: there is no impact on the contour detection. Just the response represntation gets changed
# RETR_LIST : no hierarchy, plain list
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Sort contours by area
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

#Contour Perimeter
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    #print(perimeter)

# Contour Approximation: Converts the given contour shape to another shape with lesser number of vertices
# For example: what could be the approximation of

objectContour = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approximatedShape = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    print(len(approximatedShape))

    # if our approximated contour has four points, then we can assume that we have found our receipt
    if len(approximatedShape) == 4:
        objectContour = approximatedShape
        break


print(objectContour)
# draw contours over original image. contours start from index 0. You can do any opertaion like normal array or list
# -1 means draw all contours
# 4th argument is color, last one is thickness
cv2.drawContours(image, [objectContour], -1, (0, 255, 0), 3)


(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(contour)
img = cv2.circle(img,center,radius,(0,255,0),2)