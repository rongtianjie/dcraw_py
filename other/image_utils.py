import numpy as np
import cv2
import os
import sys
import imageio
import colour

def read_tiff(infn):
    return cv2.imread(infn, cv2.IMREAD_UNCHANGED)

def imshow(img):
    if img.dtype == "uint16":
        scale = 65535
    elif img.dtype == "uint8":
        scale = 255
    elif img.dtype == "float32" or img.dtype == "float64":
        scale = 1
    else:
        print("Unknown data type.")
        return 1
    colour.plotting.plot_image(img/scale)

def save_image_16(outfn, src, verbose = False):
    imageio.imsave(outfn, src.astype(np.uint16))
    if verbose:
        print("Write file to disk [{}]".format(outfn))

def findAllSuffix(path, suffix, verbose):
    # Find all specific format of file under certain path

    # path: target path
    # suffix: file suffix e.g. ".RAF"/"RAF"
    # verbose: whether print the found path

    # return a list contain all the file paths
    
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if file.lower().endswith(suffix.lower()):
                file_path = os.path.join(root, file)
                file_path = file_path.replace("\\", "/")
                result.append(file_path)
                if verbose:
                    print(file_path)
    if verbose:
        print("Find {} [{}] files under [{}].\n".format(len(result), suffix, path))
    return result

def findRawImage(infn, fileList, suffix, verbose):
    # Import the raw data of the image with rawpy.imread()
    #
    # Input the filename and the search field path list
    # Return a rawpy object

    if "." not in infn:
        infn = infn + suffix
    infPaths = []
    for file in fileList:
        if infn in file:
            infPaths.append(file)
    if len(infPaths) == 0:
        print("Error: Cannot find [{}] under the path.".format(infn))
        sys.exit(2)
    elif len(infPaths) == 1:
        if verbose:
            print("Import [{}]\n".format(infPaths[0]))
        infPath = infPaths[0]
    else:
        for i in range(len(infPaths)):
            print("[{}]{}".format(i+1, infPaths[i]))
        n = input("\nFound Multiple files. Please choose the right image [1]-[{}]:".format(len(infPaths)))
        while int(n)-1 not in range(len(infPaths)):
            n = input("Invalid input. Please input the number from 1 to {}".format(len(infPaths)))
        if verbose:
            print("\nImport [{}]\n".format(infPaths[int(n)-1]))
        infPath = infPaths[int(n)-1]  
    return infPath

def crop_image(src, top, bottom, left, right):
    # Crop the image with margin info (2D or 3D)
    
    if len(src.shape) == 2:
        rslt = src[top : src.shape[0]-bottom, left : src.shape[1]-right].copy()
    elif len(src.shape) == 3:
        rslt = src[top : src.shape[0]-bottom, left : src.shape[1]-right, :].copy()
    else:
        print("Error: [crop_image] The input image must be in 2 or 3 dimensions.")
    return rslt

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2989, 0.587, 0.114])
    gray[gray > 65536] = 65535
    return gray.astype(np.uint16)

if __name__ == "__main__":
    print("This script contained several image basic operations.")
    exit(0)