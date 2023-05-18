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
        self.matcher = cv2.BFMatcher()
        self.draw_mathes = False
    

    def match(self, image1, image2):
        keypoints1, img1_descriptor = self.feature_extractor.detect_and_compute(image1)
        keypoints2, img2_descriptor = self.feature_extractor.detect_and_compute(image2)
        matches = self.matcher.knnMatch(img1_descriptor, img2_descriptor, k=2)

        if self.draw_mathes:
            utils.draw_matches(image1, keypoints1, image2, keypoints2, matches)

        good = []
        for m, n in matches:
            if m.distance < 0.80 * n.distance:
                good.append([m])

        matched_images = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(matched_images)
        plt.show()        


if __name__ == '__main__': 
    IMAGE1 = '2.png'
    IMAGE2 = '3.png'
    SAMPLE_IMAGE1 = os.path.join('/home/deveshdatwani/Sfm/P3Data', IMAGE1)
    SAMPLE_IMAGE2 = os.path.join('/home/deveshdatwani/Sfm/P3Data', IMAGE2)

    image1 = cv2.imread(SAMPLE_IMAGE1, cv2.IMREAD_GRAYSCALE) 
    image2 = cv2.imread(SAMPLE_IMAGE2, cv2.IMREAD_GRAYSCALE) 

    mapper = featureMapper(draw_mathes=True)
    mapper.match(image1, image2)

