import cv2
import numpy as np

img = cv2.imread('iris-1.png')
imgWidth = img.shape[1]
imgHeight = img.shape[0]



def padding(image, border_width):
    reflect = cv2.copyMakeBorder(
                image,
                border_width,border_width,border_width,border_width,
                cv2.BORDER_REFLECT)

    cv2.imshow("reflected", reflect)

    cv2.imwrite("reflected_iris.png", reflect)

# x_0: top, x_1 bottom
# y_0: left, y_1 right
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[x_0:x_1, y_0:y_1] # Slicing to crop the image

    cv2.imshow("image", image)
    cv2.imshow("cropped", cropped_image)

    cv2.imwrite("cropped_iris.png", cropped_image)


def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))

    cv2.imshow("resized image", resized_image)

    cv2.imwrite("resized_image.png", resized_image)

def copy(image, emptyPictureArray):
    cv2.imshow("image", emptyPictureArray)

    emptyPictureArray[:] = image

    cv2.imshow("emptyPictureArray", emptyPictureArray)

    cv2.imwrite("emptyPictureArray.png", emptyPictureArray)

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("gray_iris_image.png", gray_image)

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imwrite("hsv_iris_image.png", hsv_image)

def hue_shifted(image, emptyPictureArray, hue):
    cv2.imshow("image", image)

    emptyPictureArray[:] = image[:] + hue

    cv2.imshow("hue_shifted", emptyPictureArray)

    cv2.imwrite("hue_shifted_iris.png", emptyPictureArray)

def smoothing(image):
    blurred = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)

    cv2.imshow("gaussian:_blurred", blurred)
    cv2.imwrite("blurred_iris.png", blurred)

def rotation(image, rotation_angle):

    rotated = image

    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("180_degrees_iris.png", rotated)

    cv2.imshow("rotated image", rotated)


emptyPicture = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)

# == 1. Padding ==
#padding(img, 100)

## == 2. Cropping ==
#crop(img, 200, imgHeight-130, 200, imgWidth-130)

## == 3. Resize ==
#resize(img, 200, 200)

# == 4. Manual copy ==
#copy(img, emptyPicture)

# == 5. Grayscale ==
#grayscale(img)

# == 6. HSV ==
#hsv(img)

# == 7. Color shifting ==
hue_shifted(img, emptyPicture, 50)

# == 8. Smoothing ==
#smoothing(img)

# == 9. Rotation ==
#rotation(img, 180)


cv2.waitKey(0)
cv2.destroyAllWindows()

