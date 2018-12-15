import numpy as np
import csv
import pandas as pd
from PIL import Image
import time
from tempfile import TemporaryFile

"""
Data files

    training.csv: list of training 7049 images. Each row contains the (x,y) coordinates for 15 keypoints,
    and image data as row-ordered list of pixels.
     
    test.csv: list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels
    
    submissionFileFormat.csv: list of 27124 keypoints to predict. Each row contains a RowId, ImageId, FeatureName, 
    Location. FeatureName are "left_eye_center_x," "right_eyebrow_outer_end_y," etc. 
    Location is what you need to predict. 
"""


class DataLoader():

    def __init__(self, path_to_data, initialize_new=False, initialize_as_RGB=False):
        self.data = None
        self.df = self.load_data_to_df(path_to_data)

        self.images = list()
        self.keypoints = list()
        if initialize_new:
            if initialize_as_RGB:
                self.extract_images_RGB()
            else:
                self.extract_images_grayscale()

        else:
            if initialize_as_RGB:
                try:
                    self.images = np.load('../data/images_RGB.npy')
                    self.keypoints = np.load('../data/keypoints.npy')
                except FileNotFoundError:
                    self.extract_images_RGB()

            else:
                try:
                    self.images = np.load('../data/images_grayscale.npy')
                    self.keypoints = np.load('../data/keypoints.npy')

                except FileNotFoundError:
                    self.extract_images_grayscale()

        self.df = self.df.drop(['Image'], axis=1)

    def load_data_to_np(self, path, delim=",", starting_line=0):
        self.data = np.loadtxt(path, delimiter=delim, skiprows=starting_line)

    def load_data_to_df(self, path):
        return pd.read_csv(path, header=0)

    def extract_images_RGB(self):
        for index, row in self.df.iterrows():
            img = Image.new('RGB', (96, 96), "black")
            pixels = img.load()  # create the pixel map

            img_as_str = [int(i) for i in row.Image.split(' ')]
            for i in range(img.size[0]):  # for every pixel:
                for j in range(img.size[1]):
                    pixels[i, j] = (img_as_str[i + j * 96], img_as_str[i + j * 96],
                                    img_as_str[i + j * 96])  # set the colour accordingly
            self.images.append(np.array(img))

            # Create a list containing the the 30 keypoints from a row
            self.keypoints.append(np.asarray([row[x] for x in range(30)]))

        self.images = np.array(self.images)
        np.save('../data/images_RGB.npy', self.images)
        np.save('../data/keypoints.npy', self.keypoints)
        self.df.drop(['Image'], axis=1)

    def extract_images_grayscale(self):
        for index, row in self.df.iterrows():
            img = np.zeros((96, 96))
            img_as_str = [int(i) for i in row.Image.split(' ')]
            for i in range(img.shape[0]):  # for every pixel:
                for j in range(img.shape[1]):
                    img[i][j] = img_as_str[i + j * 96]
            self.images.append(np.array(img))
            # Create a list containing the the 30 keypoints from a row
            self.keypoints.append(np.asarray([row[x] for x in range(30)]))

        self.images = np.array(self.images)
        np.save('../data/images_grayscale.npy', self.images)
        np.save('../data/keypoints.npy', self.keypoints)
        self.df.drop(['Image'], axis=1)


#dl = DataLoader("../data/training.csv", initialize_new=True)

test = 0
