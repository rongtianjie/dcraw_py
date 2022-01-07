# This script contains camera data
from typing import NoReturn
import numpy as np

class camData:
    def __init__(self):
        self.black_level_per_channel = []
        self.camera_white_level_per_channel = None
        self.camera_whitebalance = []
        self.color_desc = ""
        self.color_matrix = np.zeros((3, 4))
        self.daylight_whitebalance = []
        self.num_colors = 0
        self.raw_pattern = None
        self.raw_type = None
        self.rgb_xyz_matrix = np.zeros((4, 3))
        self.sizes = None
        self.tone_curve = np.zeros(65536)
        self.white_level = 0

class ImageSizes:
    def __init__(self, raw_height, raw_width, height, width, top_margin, left_margin, iheight, iwidth, pixel_aspect, flip):
        self.raw_height = raw_height
        self.raw_width = raw_width
        self.height = height
        self.width = width
        self.top_margin = top_margin
        self.left_margin = left_margin
        self.iheight = iheight
        self.iwidth = iwidth
        self.pixel_aspect = pixel_aspect
        self.flip = flip


def GFX100S():
    data = camData()
    data.black_level_per_channel = [255, 255, 255, 255]
    # white balance data has to be read from image
    data.camera_whitebalance = [599.0, 302.0, 443.0, 0.0]
    data.color_desc = b'RGBG'
    data.daylight_whitebalance = [1.9002734422683716, 0.945999801158905, 1.3716827630996704, 0.0]
    data.num_colors = 3
    data.raw_pattern = [[0, 1], [3, 2]]
    data.rgb_xyz_matrix = [[1.6212, -0.8423, -0.1583], [-0.4336,  1.2583, 0.1937], [-0.0195, 0.0726, 0.6199], [0, 0, 0]]
    data.tone_curve = list(range(65536))
    data.white_level = 15872
    data.sizes = ImageSizes(8754, 11808, 8752, 11662, 2, 0, 8752, 11662, 1.0, 0)
    return data

# Crop information
# Used to crop the image from libraw size to official size
# GFX100S official image size is 11648 x 8736 (libraw: 11662 x 8752)

class Crop_DB:
    def __init__(self, top_margin, bottom_margin, left_margin, right_margin):
        self.top = top_margin
        self.bottom = bottom_margin
        self.left = left_margin
        self.right = right_margin

def CROP_GFX100S():
    return Crop_DB(5, 11, 8, 6)