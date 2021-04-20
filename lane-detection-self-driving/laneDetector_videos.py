import cv2
import numpy as np

def canyEdgeDetector(image):
    edged = cv2.Canny(image, 50, 150)
    return edged


def getROI(image):
    height = image.shape[0]
    width = image.shape[1]
    # Defining Triangular ROI: The values will change as per your camera mounts
    triangle=np.array([[(200, height), (1100, height), (550, 250)]])
    # creating black image same as that of input image
    black_image = np.zeros_like(image)
    # Put the Triangular shape on top of our Black image to create a mask
    mask = cv2.fillPoly(black_image, triangle, 255)
    # applying mask on original image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



def getLines(image):
    # lines=cv2.HoughLinesP(image,bin_size,precision,threshold,dummy 2d array--no use,minLineLength,maxLineGap)
    # lets take bin size to be 2 pixels
    # lets take precision to be 1 degree= pi/180 radians
    # threshold is the votes that a bin should have to be accepted to draw a line
    # minLineLength --the minimum length in pixels a line should have to be accepted.
    # maxLineGap --the max gap between 2 broken line which we allow for 2 lines to be connected together.
    lines = cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    return lines


#display lines over a image
def displayLines(image, lines):
    if lines is not None:
        for line in lines:
            # print(line) --output like [[704 418 927 641]] this is 2d array representing [[x1,y1,x2,y2]] for each line
            x1, y1, x2, y2 = line.reshape(4)  # converting to 1d array []

            # draw line over black image --(255,0,0) tells we want to draw blue line (b,g,r) values 10 is line thickness
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return image



def getLineCoordinatesFromParameters(image, line_parameters):
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = image.shape[0]  # since line will always start from bottom of image
    y2 = int(y1 * (3.4 / 5))  # some random point at 3/5
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])



#Avergaes all the left and right lines found for a lane and retuns single left and right line for the the lane
def getSmoothLines(image, lines):
    left_fit = []  # will hold m,c parameters for left side lines
    right_fit = []  # will hold m,c parameters for right side lines

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # polyfit gives slope(m) and intercept(c) values from input points
        # last parameter 1 is for linear..so it will give linear parameters m,c
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # take averages of all intercepts and slopes separately and get 1 single value for slope,intercept
    # axis=0 means vertically...see its always (row,column)...so row is always 0 position.
    # so axis 0 means over row(vertically)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # now we have got m,c parameters for left and right line, we need to know x1,y1 x2,y2 parameters
    left_line = getLineCoordinatesFromParameters(image, left_fit_average)
    right_line = getLineCoordinatesFromParameters(image, right_fit_average)
    return np.array([left_line, right_line])






videoFeed = cv2.VideoCapture("test_video.mp4")

try:
  while videoFeed.isOpened() :
    (status, image) = videoFeed.read()

    edged_image = canyEdgeDetector(image)   # Step 1
    roi_image = getROI(edged_image)         # Step 2

    lines = getLines(roi_image)             # Step 3
    #image_with_lines = displayLines(image, lines)

    smooth_lines = getSmoothLines(image, lines)    # Step 5
    image_with_smooth_lines = displayLines(image, smooth_lines) # Step 4

    cv2.imshow("Output", image_with_smooth_lines)
    cv2.waitKey(1)

except:
    pass