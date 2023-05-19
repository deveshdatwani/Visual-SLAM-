import cv2
import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from matplotlib import pyplot as plt


def get_k():
    K_ADDRESS = '/home/deveshdatwani/Sfm/P3Data/calibration.txt'
    k_params = np.loadtxt(K_ADDRESS)

    return k_params

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


def compute_fundamental_matrix(pts1, pts2):
    A = np.zeros(shape=(8,9))

    x = pts1[:,0]
    y = pts1[:,1]
    x_dash = pts2[:,0]
    y_dash = pts2[:,1]

    for i in range(8):
        A[i] = np.array([x[i]*x_dash[i], x[i]*y_dash[i], x[i], y[i]*x_dash[i], y[i]*y_dash[i], y[i], x_dash[i], y_dash[i], 1])

    U, S, V = svd(A, full_matrices=True)
    F = V.transpose()[:,-1].reshape((3,3))
    
    return F


def fundamental_matrix(keypoints1, keypoints2, matches, N):
    points1 = []
    points2 = []

    for match in matches[:N]:
        points1.append(keypoints1[match.queryIdx].pt)
        points2.append(keypoints2[match.imgIdx].pt)

    points1, points2 = np.float32(points1), np.float32(points2)

    # NEED TO IMPLEMENT RANSAC

    pts1, pts2 = points1[np.random.randint(0, 100, 8)], points2[np.random.randint(0, 100, 8)]
    F = compute_fundamental_matrix(pts1, pts2)

    return F


def essential_matrix(F):
    k = get_k()
    E = np.dot(np.dot(k.T, F), k)

    return E 


def normalize_coordinates(pts1, pts2):
    X1 = pts1[:,0]
    Y1 = pts1[:,1]
    X2 = pts2[:,0]
    Y2 = pts2[:,1]

    X1_MEAN = X1.mean()
    Y1_MEAN = Y1.mean()
    X2_MEAN = X2.mean()
    Y2_MEAN = Y2.mean()

    X1_STD = X1.std()
    Y1_STD = Y1.std()
    X2_STD = X2.std()
    Y2_STD = Y2.std()

    VAR_1 = np.array([[X1_STD, 0,0 ],[0, Y1_STD, 0],[0,0,1]])
    MEAN_1 = np.array([[1,0,-X1_MEAN], [0,1,-Y1_MEAN], [0,0,1]])

    pts1 = np.column_stack((pts1, np.ones(shape=(len(pts1)))))
    NORMAL_PTS1 = np.dot(np.dot(VAR_1, MEAN_1), pts1.T)

    VAR_2 = np.array([[X2_STD, 0,0 ],[0, Y2_STD, 0],[0,0,1]])
    MEAN_2 = np.array([[1,0,-X2_MEAN], [0,1,-Y2_MEAN], [0,0,1]])

    pts2 = np.column_stack((pts2, np.ones(shape=(len(pts2)))))
    NORMAL_PTS2 = np.dot(np.dot(VAR_2, MEAN_2), pts2.T)

    print(NORMAL_PTS1.T.shape)
    print(NORMAL_PTS2.T.shape)

    return NORMAL_PTS1.T, NORMAL_PTS2.T