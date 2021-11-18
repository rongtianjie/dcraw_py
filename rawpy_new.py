import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import rawpy
import rawpy.enhance
import imageio
import time
# from PIL import Image
import cv2
import os
import colour_demosaicing
import math
import random
import sys
import getopt


# Define margin for raw data
# Actural effiective resolution is 11664 x 8749
top_margin = 1
bottom_margin = 4
left_margin = 0
right_margin = 144

# Define default pixel max value
maximum = 65535

def FindAllSuffix(path, suffix, verbose = False):
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
    print("Find {} [{}] files under [{}].\n".format(len(result), suffix, path))
    return result

def importRawImage(infn, fileList, suffix, verbose = False):
    # Import the raw data of the image with rawpy.imread()

    # return a rawpy object

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
        return rawpy.imread(infPaths[0])
    else:
        for i in range(len(infPaths)):
            print("[{}]{}".format(i+1, infPaths[i]))
        n = input("\nFound Multiple files. Please choose the right image [1]-[{}]:".format(len(infPaths)))
        while int(n)-1 not in range(len(infPaths)):
            n = input("Invalid input. Please input the number from 1 to {}".format(len(infPaths)))
        if verbose:
            print("\nImport [{}]\n".format(infPaths[int(n)-1]))
        return rawpy.imread(infPaths[int(n)-1])   

def bad_fix(fileList, rawData, verbose = False):
    # Fix bad pixels with rawpy.enhance
    if len(fileList) >= 1:
        if len(fileList) <= 10:
            sample_num = len(fileList)
        else:
            sample_num = min(10, len(fileList)//2)

        if verbose:
            print("Starting bad pixel fixing.")

        blc_sample = random.sample(fileList, sample_num)

        if verbose:
            print("Finding bad pixel using {} images...".format(sample_num))
            
        bad_list = rawpy.enhance.find_bad_pixels(blc_sample, find_hot=True, find_dead=False, confirm_ratio=0.9)
        
        if verbose:
            print("Found {} bad pixels:".format(len(bad_list)))
            # print(bad_list)
        
        rawpy.enhance.repair_bad_pixels(rawData, bad_list, method='median')
        if verbose:
            print("Bad pixel fixing finished.\n")
    return rawData

def crop_image(src, top, bottom, left, right):
    # Crop the image with margin info (2D or 3D)
    
    if len(src.shape) == 2:
        dst = src[top : src.shape[0]-bottom, left : src.shape[1]-right].copy()
    elif len(src.shape) == 3:
        dst = src[top : src.shape[0]-bottom, left : src.shape[1]-right, :].copy()
    else:
        print("Error: [crop_image] The input image must be in 2 or 3 dimensions.")
    return dst

def adjust_maximum(raw, maximum_thr = 0.75):
    global maximum
    real_max = raw.raw_image_visible.max()
    if real_max > 0 and real_max < maximum and real_max > maximum * maximum_thr:
        maximum = real_max

def subtraction(raw, USE_MIN_BLC = 0):
    # BLC on raw image pattern
    # Input should be rawpy object

    # Output will be crop by rawpy "_visible" 
    # On GFX100S, image size is 8752 * 11662

    dst = raw.raw_image_visible.astype(np.float64)

    # if USE_MIN_BLC:
    #     sort_level = np.sort(raw.raw_image_visible)
    #     if sort_level[0] == 0:
    #         bl = sort_level[0]
    #     else:
    #         bl = sort_level[1]
    #         dst -= bl
    # else:
    for i, bl in enumerate(raw.black_level_per_channel):
        dst[raw.raw_colors_visible == i] -= bl

    dst = CLIP(dst)
    return dst


def scale_colors(raw, verbose = False):
    src_blc = subtraction(raw)


    if verbose:
        print("Start white balance correction with camera setting.")
    wb = raw.camera_whitebalance
    wb_coeff = np.asarray(wb[:3]) / max(wb[:3])
    wb_coeff = np.append(wb_coeff,wb_coeff[1])

    if verbose:
        print("WB coefficient is {}".format(wb_coeff))

    if raw.camera_white_level_per_channel is None:
        white_level = [maximum] * 4
    else:
        white_level = raw.camera_white_level_per_channel
    
    white_level = np.array(white_level) - np.array(raw.black_level_per_channel)

    scale_coeff = wb_coeff * 65535 / white_level
    print("Scale coefficient is {}".format(scale_coeff))

    scale_matrix = np.empty(raw.raw_colors_visible.shape, dtype=np.float64)

    for i, scale_co in  enumerate(scale_coeff):
        scale_matrix[raw.raw_colors_visible == i] = scale_co
    
    dst = CLIP(src_blc * scale_matrix)
    
    if verbose:
        print("White balance finished.\n")

    return dst


def demosaicing(src, Bayer_Pattern, DEMOSACING_METHOD = 0, verbose = False):
    # Demosaicing. Default using bilinear
    if verbose:
        print("Start demosaicing.")

    numbers = {
    0 : colour_demosaicing.demosaicing_CFA_Bayer_bilinear, 
    1 : colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004, 
    2 : colour_demosaicing.demosaicing_CFA_Bayer_Menon2007
    # Menon2007 needs more than 20 GB memory
    }

    method = numbers.get(DEMOSACING_METHOD, colour_demosaicing.demosaicing_CFA_Bayer_bilinear )

    if verbose:
        print("Demosacing using [{}]...".format(method))
    dst = method(src, Bayer_Pattern)

    if verbose:
        print("Demosacing finished.\n")

    return dst

def auto_bright(src):
    dst = src * 4.9 + 1000
    dst[dst>65535] = 65535
    return dst

def CLIP(src):
    dst = src.copy()
    dst[dst>65536] = 65535
    dst[dst<0] = 0
    return dst



if __name__ == "__main__":
    print("Start script...\n")

    path = sys.path[0]+"/images"
    infn = "DSCF3396.RAF"
    outfn = "demosaiced.tiff"
    suffix = ".RAF"
    verbose = True

    opts, args = getopt.getopt(sys.argv[1:], "-h-i:-p:-o:-v",["help","path=","ifile=","ofile="])

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("rawpy_new.py -p <ImgPath> -i <InputFile> -o <OutputFile> -v")
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-i", "--ifile"):
            infn = arg
        elif opt in ("-o", "--ofile"):
            outfn = arg
        elif opt in ("-v"):
            verbose = True
    
    if (infn == "" or outfn == ""):
        print("Error: rawpy_new.py -p <ImgPath> -i <InputFile> -o <OutputFile>")
        sys.exit(2)

    fileList = FindAllSuffix(path, suffix, verbose)

    rawData = importRawImage(infn, fileList, suffix, verbose)
    # imageio.imsave("raw.tiff", rawData.raw_image)

    rawData_badfix = bad_fix(fileList, rawData, verbose)

    adjust_maximum(rawData_badfix)
    
    rawImage_wb = scale_colors(rawData_badfix, verbose)

    # image_demosaiced = demosaicing(rawImage_wb, "RGGB", 2, verbose)
    image_demosaiced = demosaicing(rawData_badfix.raw_image_visible, "RGGB", 2, verbose)


    print(image_demosaiced.shape)
    red_avg = image_demosaiced[:,:,0].mean()
    green_avg = image_demosaiced[:,:,1].mean()
    blue_avg = image_demosaiced[:,:,2].mean()
    print(red_avg/green_avg, 1, blue_avg/green_avg)

    imageio.imsave("demosaic.tiff", image_demosaiced.astype(np.uint16))

    reference = rawData.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb = True)
    red_avg_ref = reference[:,:,0].mean()
    green_avg_ref = reference[:,:,1].mean()
    blue_avg_ref = reference[:,:,2].mean()
    print(red_avg_ref/green_avg_ref, 1, blue_avg_ref/green_avg_ref)
    imageio.imsave("ref.tiff", reference.astype(np.uint16))

    