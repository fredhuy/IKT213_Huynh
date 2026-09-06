import cv2
import numpy as np

img = cv2.imread('images/shapes-1.png')
template = cv2.imread('images/shapes_template.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

def template_match(image, template):
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9

    w, h = template.shape[::-1]

    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    cv2.imwrite('images/res.png', image)


template_match(img_gray, template_gray)