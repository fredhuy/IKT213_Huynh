import cv2
import numpy as np


img = cv2.imread('images/lambo.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Display original image

def sobel_edge_detection(image):

    img_gaussian = cv2.GaussianBlur(image, (3, 3), 0)

    sobely = cv2.Sobel(src=img_gaussian, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)

    sobelx = cv2.Sobel(src=img_gaussian, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)

    sobelxy = cv2.magnitude(sobelx, sobely)

    sobelFinalized = cv2.normalize(
        sobelxy,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U)


    cv2.imshow('Sobel gradient magnitude', sobelFinalized)
    cv2.imwrite('images/sobel_lambo.png', sobelx)

    cv2.waitKey(0)


def canny_edge_detection(image, threshold_1, threshold_2):
    img_gaussian = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(image=img_gaussian, threshold1=threshold_1, threshold2=threshold_2)

    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)






#sobel_edge_detection(img_gray)

canny_edge_detection(img_gray, 100, 150)