import matplotlib.pyplot as plt
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
def detect(c):
    shape = None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
    elif len(approx) == 2:
        shape='not identified'
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
        # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
        # return the name of the shape
    return shape
def Count_shapes(image_path):
    image = cv2.imread(image_path)
    print('Showing original image')
    plt.imshow(image)
    plt.show()
    resized = imutils.resize(image, width=450)
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret,thresh = cv2.threshold(blurred, 215, 240, cv2.THRESH_BINARY)
    # find contours in the thresholded image and initialize the
    # shape detector
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    print('Displaying Contoures of images')
    plt.imshow(img)
    plt.show()

    shapes=[]
    for c in contours:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = detect(c)
        shapes.append(shape)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 0), 2)
        # show the output image
    print('Displaying number of shapes')
    for cnt in set(shapes):
        print(shapes.count(cnt),cnt)
    plt.imshow(image)
    plt.show()

Count_shapes('ct.jpg')
