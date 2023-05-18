import cv2
from scipy.linalg import svd
from matplotlib import pyplot as plt


def draw_key_points(image, keypoints):
    img2 = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=0)
    plt.imshow(img2)
    plt.show()

    return None


def draw_matches(image1, image2, keypoints1, keypoints2, matches):
    matched_images = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_images)
    plt.show()

    return None


def fundamental_matrix(points1, points2):
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)
    return None

