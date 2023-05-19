import os
import cv2
import utils  
import numpy as np  
from matplotlib import pyplot as plt
from feature_mapping import featureMapper
from feature_extraction import featureExtractor


if __name__ == '__main__': 
    IMAGE1 = '1.png'
    IMAGE2 = '2.png'
    
    SAMPLE_IMAGE1 = os.path.join('/home/deveshdatwani/Sfm/P3Data', IMAGE1)
    SAMPLE_IMAGE2 = os.path.join('/home/deveshdatwani/Sfm/P3Data', IMAGE2)

    image1 = cv2.imread(SAMPLE_IMAGE1, cv2.IMREAD_GRAYSCALE) 
    image2 = cv2.imread(SAMPLE_IMAGE2, cv2.IMREAD_GRAYSCALE) 

    mapper = featureMapper(draw_mathes=False)
    keypoints1, keypoints2, matches = mapper.match(image1, image2)

    F = utils.fundamental_matrix(keypoints1, keypoints2, matches, 200)

    keypoints1, keypoints2 = np.array([i.pt for i in keypoints1]), np.array([j.pt for j in keypoints2])

    K = utils.get_k()
    E = utils.essential_matrix(F)

    normalized_keypoints1, normalized_keypoints = utils.normalize_coordinates(keypoints1, keypoints2)

