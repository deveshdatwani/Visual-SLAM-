import os
import cv2 
import numpy as np
from utils import *
from matplotlib import pyplot as plt


class featureExtractor():
    def __init__(self, extractor_type='ORB', n_keypoints=1000):
        self.n_keypoints = n_keypoints

        if extractor_type == 'ORB':
            self.extractor = cv2.ORB_create(nfeatures=n_keypoints)
            self.n_keypoints = n_keypoints


    def detect_keypoints(self, image):
        assert len(image.shape) > 1 
        keypoints = self.extractor.detect(image)
        print(f'{self.n_keypoints} detected by the detector')
        responses = [i.response for i in keypoints]
        
        return keypoints


    def extract_features(self, image, keypoints):
        keypoints, descriptors = self.extractor.compute(image, keypoints)

        return keypoints, descriptors


    def detect_and_compute(self, image):
        assert len(image.shape) > 1 
        keypoints = self.detect_keypoints(image)
        keypoints, features = self.extract_features(image, keypoints)

        return keypoints, features
    
