import numpy as np
import cv2
from matplotlib import pyplot as plt
# mpl.rcParams['figure.figsize'] = 50, 25
import math
import sys
import dcraw_utils
import image_utils

def image_hist(image):
    # 3-channel RGB image
    color = {"red", "green", "blue"}
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [image.max()], [0, image.max()])
        plt.figure()
        plt.plot(hist, color=color)
        plt.xlim([0, image.max()])
    plt.show()

def image_comp(path1, path2):
    output1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
    output2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

    if output1.shape == output2.shape:
        diff = cv2.absdiff(output1,output2)
        print(np.mean(diff))

        mse = np.mean(diff.astype(np.int32) ** 2)
        print("MSE: {}".format(mse))
        psnr = 10 * math.log10((65535**2) / mse)
        print("PSNR is {}".format(psnr))
        print("Maximum difference in 16-bit is {} which is located at {}".format(diff.max(), np.argwhere(diff.max() == diff)))

        image_hist(diff)
            
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow((diff/257).astype(np.uint8))
            
        # Normalize diff matrix for visualization
        diff_norm = (diff - diff.min())/(diff.max()-diff.min())
        ax = fig.add_subplot(212)
        ax.imshow(diff_norm)
    else:
        print("size does not match!")

def channel_avg(img):
    red_avg = img[:,:,0].mean()
    green_avg = img[:,:,1].mean()
    blue_avg = img[:,:,2].mean()
    print("[", red_avg/green_avg, 1, blue_avg/green_avg, "]")

def raw_info(raw):
    print("black_level_per_channel")
    print(raw.black_level_per_channel)

    print("camera_white_level_per_channel")
    print(raw.camera_white_level_per_channel)

    print("camera_whitebalance")
    print(raw.camera_whitebalance)

    print("color_desc")
    print(raw.color_desc)

    print("color_matrix")
    print(raw.color_matrix)

    print("daylight_whitebalance")
    print(raw.daylight_whitebalance)

    print("num_colors")
    print(raw.num_colors)

    print("raw_colors")
    print(raw.raw_colors)

    print("raw_colors_visible")
    print(raw.raw_colors_visible)

    print("raw_pattern")
    print(raw.raw_pattern)

    print("raw_type")
    print(raw.raw_type)

    print("rgb_xyz_matrix")
    print(raw.rgb_xyz_matrix)

    print("tone_curve")
    print(raw.tone_curve)

    print("white_level")
    print(raw.white_level)

    print("max()")
    print(raw.raw_image.max())

    print("sizes")
    print(raw.sizes)

    print("sizes1")
    print(raw.raw_image_visible.shape)

if __name__ == "__main__":
    infn = "DSCF3396.RAF"
    suffix = ".RAF"

    if infn == "":
        infn = sys.argv[1]

    fileList = image_utils.FindAllSuffix(sys.path[0]+"/images", suffix)
    infPath = image_utils.findRawImage(infn, fileList, suffix)
    rawData = dcraw_utils.importRawImage(infPath)

    raw_info(rawData)