import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = 50, 25
import math
import sys 

def image_hist(image):
    color = {"red", "green", "blue"}
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [diff.max()], [0, diff.max()])
        plt.plot(hist, color=color)
        plt.xlim([0, diff.max()])
    plt.show()

path1 = sys.argv[1]
path2 = sys.argv[2]

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

