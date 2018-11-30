import numpy as np
import csv
import pandas as pd
from PIL import Image
import time
from tempfile import TemporaryFile


class Sample():
    def __init__(self, series):
        self.left_eye_center = (series.left_eye_center_x, series.left_eye_center_y)
        self.right_eye_center = (series.right_eye_center_x, series.right_eye_center_y)
        self.left_eye_inner_corner = (series.left_eye_inner_corner_x, series.left_eye_inner_corner_y)
        self.left_eye_outer_corner = (series.left_eye_outer_corner_x, series.left_eye_outer_corner_y)
        self.right_eye_inner_corner = (series.right_eye_inner_corner_x, series.right_eye_inner_corner_y)
        self.right_eye_outer_corner = (series.right_eye_outer_corner_x, series.right_eye_outer_corner_y)
        self.left_eyebrow_inner_end = (series.left_eyebrow_inner_end_x, series.left_eyebrow_inner_end_y)
        self.left_eyebrow_outer_end = (series.left_eyebrow_outer_end_x, series.left_eyebrow_outer_end_y)
        self.right_eyebrow_inner_end = (series.right_eyebrow_inner_end_x, series.right_eyebrow_inner_end_y)
        self.right_eyebrow_outer_end = (series.right_eyebrow_outer_end_x, series.right_eyebrow_outer_end_x)
        self.nose_tip = (series.nose_tip_x, series.nose_tip_y)
        self.mouth_left_corner = (series.mouth_left_corner_x, series.mouth_left_corner_y)
        self.mouth_right_corner = (series.mouth_right_corner_x, series.mouth_right_corner_y)
        self.mouth_center_top_lip = (series.mouth_center_top_lip_x, series.mouth_center_top_lip_y)
        self.mouth_center_bottom_lip = (series.mouth_center_bottom_lip_x, series.mouth_center_bottom_lip_y)
        print (self.left_eye_center)


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

    def __init__(self, path_to_data, initialize_new=False):
        self.data = None
        self.df = self.load_data_to_df(path_to_data)

        self.images = None
        if initialize_new:
            self.extract_images()
        else:
            self.images = np.load('images.npy')
        self.df = self.df.drop(['Image'], axis=1)
        self.first_sample = Sample(self.df.iloc[1])

    def load_data_to_np(self, path, delim=",", starting_line=0):
        self.data = np.loadtxt(path, delimiter=delim, skiprows=starting_line)

    def load_data_to_df(self, path):
        return pd.read_csv(path, header=0)

    def extract_images(self):
        for index, row in self.df.iterrows():
            img = Image.new('RGB', (96, 96), "black")
            pixels = img.load()  # create the pixel map

            img_as_str = [int(i) for i in row.Image.split(' ')]
            for i in range(img.size[0]):  # for every pixel:
                for j in range(img.size[1]):
                    pixels[i, j] = (img_as_str[i + j * 96], img_as_str[i + j * 96],
                                    img_as_str[i + j * 96])  # set the colour accordingly

            self.images.append(np.array(img))

        self.images = np.array(self.images)
        np.save('images.npy', self.images)
        self.df.drop(['Image'], axis=1)
        test = 0


dl = DataLoader("../data/training.csv")

test = 0
