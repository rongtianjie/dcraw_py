# All operation will be in dtype = float64

import numpy as np
import rawpy
import rawpy.enhance
import imageio
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

xyz_srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]])

def read_tiff(infn):
    return cv2.imread(infn, cv2.IMREAD_UNCHANGED)

def save_image_16(outfn, src):
    imageio.imsave(outfn, src.astype(np.uint16))

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
    return rawpy.imread(infPath), infPath

def bad_fix(fileList, rawData, verbose = False):
    # Fix bad pixels with rawpy.enhance
    # 
    # Input the rawpy object and image list
    # return a rawpy object with bad pixel fixed 
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
        rslt = src[top : src.shape[0]-bottom, left : src.shape[1]-right].copy()
    elif len(src.shape) == 3:
        rslt = src[top : src.shape[0]-bottom, left : src.shape[1]-right, :].copy()
    else:
        print("Error: [crop_image] The input image must be in 2 or 3 dimensions.")
    return rslt

def subtract(raw, dark_img, fileList, verbose = False):
    # subtract dark frame to remove noise floor
    # Input: bayer pattern image, dark frame filename,
    if verbose:
        print("Subtraction using dark frame...")

    darkData, infPath = importRawImage(dark_img, fileList, ".RAF", verbose)
    darkData_badfix = bad_fix([infPath], darkData, verbose)
    noise_floor = darkData_badfix.raw_image_visible.max()

    if verbose:
        print("The noise floor is {}\n".format(noise_floor))

    rslt = raw.raw_image_visible.astype(np.int32) - noise_floor
    return CLIP(rslt)

def blc(raw, USE_MIN_BLC = 0):
    # BLC on raw image pattern
    # Input should be rawpy object

    # Output will be crop by rawpy "_visible" 
    # On GFX100S, image size is 8752 * 11662
    
    rslt = raw.raw_image_visible.astype(np.int32)

    # if USE_MIN_BLC:
    #     sort_level = np.sort(raw.raw_image_visible)
    #     if sort_level[0] == 0:
    #         bl = sort_level[0]
    #     else:
    #         bl = sort_level[1]
    #         rslt -= bl
    # else:
    for i, bl in enumerate(raw.black_level_per_channel):
        rslt[raw.raw_colors_visible == i] -= bl

    return CLIP(rslt)

def adjust_maximum(raw, maximum_thr = 0.75):
    global maximum
    real_max = raw.raw_image_visible.max()
    if real_max > 0 and real_max < maximum and real_max > maximum * maximum_thr:
        maximum = real_max

def scale_colors(src, raw, verbose = False):
    if src==None or src.shape != raw.raw_image.shape:
        src_blc = blc(raw)

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
    
    rslt = CLIP(src_blc * scale_matrix)
    
    if verbose:
        print("White balance finished.\n")

    return rslt


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
    rslt = method(src, Bayer_Pattern)

    if verbose:
        print("Demosacing finished.\n")

    return rslt

def auto_bright(src):
    rslt = src * 4.9 + 1000
    rslt[rslt>65535] = 65535
    return rslt

def CLIP(src):
    rslt = src.copy()
    rslt[rslt>65536] = 65535
    rslt[rslt<0] = 0
    return rslt
    
def camera_to_srgb(src, raw, verbose = False):
    shape = src.shape
    if shape[2] != 3:
        print("The input image should be 3-channel.")
        exit(1)
    else:
        cam = src.reshape((shape[0]*shape[1], shape[2]))
        xyz_2d = np.dot(raw.rgb_xyz_matrix[:3][:], cam.T).T
        srgb_2d = np.dot(xyz_srgb, xyz_2d.T).T

        image_srgb = srgb_2d.reshape(shape)
    return image_srgb

def color_check_correction():
    return 0 

if __name__ == "__main__":

    path = sys.path[0]
    infn = ""
    if "." in infn:
        outfn = path + infn.split(".", 1)[0]
    suffix = ".RAF"
    verbose = True
    USE_DARK = False

    opts, args = getopt.getopt(sys.argv[1:], "-h-i:-p:-o:-v-K:",["help","path=","ifile=","ofile=","dark="])

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            man_file = sys.path[0] + "/manual"
            if os.path.isfile(man_file):
                file_object = open(man_file)
                manual = file_object.read()
                print(manual)
            else:
                print("Can not find the manual. Please refer to the script.")
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-i", "--ifile"):
            infn = arg
        elif opt in ("-o", "--ofile"):
            outfn += "_" + arg + ".tiff"
        elif opt in ("-v"):
            verbose = True
        elif opt in ("-K", "--dark"):
            dark_frame = arg
            USE_DARK = True
    
    if (infn == ""):
        print("Error: dcraw_utils.py -p <ImgPath> -i <InputFile> -o <OutputFile> -K <Dark frame>")
        sys.exit(2)

    fileList = FindAllSuffix(path, suffix, verbose)

    rawData, __ = importRawImage(infn, fileList, suffix, verbose)

    rawData_badfix = bad_fix(fileList, rawData, verbose)

    if USE_DARK:
        img_sub = subtract(rawData_badfix, dark_frame, fileList, verbose)
        scale_in = img_sub
    else:
        scale_in = None

    adjust_maximum(rawData_badfix)
    rawImage_wb = scale_colors(scale_in, rawData_badfix, verbose)

    image_demosaiced = demosaicing(rawImage_wb, "RGGB", 0, verbose)

    save_image_16(outfn, image_demosaiced)
    