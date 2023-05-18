import os
import cv2
import utils  
import numpy as np  
from matplotlib import pyplot as plt
from feature_mapping import featureMapper
from feature_extraction import featureExtractor


if __name__ == '__main__': 
    IMAGE1 = '3.png'
    IMAGE2 = '4.png'
    
    SAMPLE_IMAGE1 = os.path.join('/home/deveshdatwani/Sfm/P3Data', IMAGE1)
    SAMPLE_IMAGE2 = os.path.join('/home/deveshdatwani/Sfm/P3Data', IMAGE2)

    image1 = cv2.imread(SAMPLE_IMAGE1, cv2.IMREAD_GRAYSCALE) 
    image2 = cv2.imread(SAMPLE_IMAGE2, cv2.IMREAD_GRAYSCALE) 

    mapper = featureMapper(draw_mathes=False)
    keypoints1, keypoints2, matches = mapper.match(image1, image2)

    one_match = matches[np.random.randint(100)]

    pt1 = keypoints1[one_match.trainIdx].pt
    pt2 = keypoints2[one_match.queryIdx].pt

    
