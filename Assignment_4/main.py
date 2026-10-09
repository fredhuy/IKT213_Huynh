import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./images/reference_img.png")

GOOD_MATCH_PERCENT = 0.7
MAX_FEATURES = 10

img1 = cv2.imread('Images/reference_img-1.png', cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('Images/align_this.jpg', cv2.IMREAD_GRAYSCALE) # trainImage

def harris_corner_detection(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # big corners
    dst = cv2.dilate(dst, None)

    #threshold
    reference_image[dst > 0.1 * dst.max()] = [0, 0, 255]

    cv2.imwrite("SavedImages/harris_corner_detection.png", reference_image)

def sift(image_to_align, reference_image, max_features, good_match_precent):

    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kp2, des2 = sift.detectAndCompute(image_to_align, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches
    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = reference_image.shape

        aligned = cv2.warpPerspective(
            image_to_align,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )

        cv2.imwrite("SavedImages/aligned.png", aligned)

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        image_to_align = cv2.polylines(image_to_align, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    final = cv2.drawMatches(reference_image, kp1, image_to_align, kp2, good, None, **draw_params)

    cv2.imwrite("SavedImages/sift.png", final)

def main():
    harris_corner_detection(img)
    sift(img2, img1, MAX_FEATURES, GOOD_MATCH_PERCENT)


main()
