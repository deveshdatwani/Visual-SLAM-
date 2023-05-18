import os
import cv2 
import numpy as np
import utils 
from matplotlib import pyplot as plt
from feature_extraction import featureExtractor


class featureMapper():
    def __init__(self, algorithm='FLANN', draw_mathes=False):
        self.algorithm = algorithm
        self.feature_extractor = featureExtractor()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.draw_mathes = draw_mathes
    

    def match(self, image1, image2):
        keypoints1, img1_descriptor = self.feature_extractor.detect_and_compute(image1)
        keypoints2, img2_descriptor = self.feature_extractor.detect_and_compute(image2)
        matches = self.matcher.match(img1_descriptor, img2_descriptor)

        keypoints1 = np.asarray(keypoints1)
        keypoints2 = np.asarray(keypoints2)
        matches = sorted(matches, key = lambda x:x.distance)

        if self.draw_mathes:
            matched_images = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
            plt.imshow(matched_images)
            plt.show()   
            
        return keypoints1, keypoints2, matches     


