import numpy as np
import rawpy
import os
import imageio
import sys

path = "."
im_suffix = ".RAF"

# Define margin for raw data
# Actural effiective resolution is 11664 x 8749
top_margin = 1
bottom_margin = 4
left_margin = 0
right_margin = 144
# ImageSizes(raw_height=8754, raw_width=11808, height=8752, width=11662, top_margin=2, left_margin=0, iheight=8752, iwidth=11662, pixel_aspect=1.0, flip=0)

def FindAllSuffix(path, suffix, verbose = False):
    # Find all specific format of file under certain path

    # path: target path
    # suffix: file suffix e.g. ".RAF"/"RAF"
    # verbose: whether print the found path
    
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if suffix or suffix.swapcase() in file:
                file_path = os.path.join(root, file)
                file_path = file_path.replace("\\", "/")
                result.append(file_path)
                if verbose:
                    print(file_path)
    if verbose:
        print("Find {} [{}] files under [{}].\n".format(len(result), suffix, path))
    return result

def importRawImage(infn, fileList, verbose = False):
    # Import the raw data of the image with rawpy.imread().
    if "." not in infn:
        infn = infn + im_suffix
    infPaths = []
    for file in fileList:
        if infn in file:
            infPaths.append(file)
    if len(infPaths) == 0:
        if verbose:
            print("Error: Cannot find [{}] under the path.".format(infn))
        return -1
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
    
infn = sys.argv[1]
fileList = FindAllSuffix(path, ".RAF")
rawData = importRawImage(infn, fileList)

print("black_level_per_channel")
print(rawData.black_level_per_channel)

print("camera_white_level_per_channel")
print(rawData.camera_white_level_per_channel)

print("camera_whitebalance")
print(rawData.camera_whitebalance)

print("color_desc")
print(rawData.color_desc)

print("color_matrix")
print(rawData.color_matrix)

print("daylight_whitebalance")
print(rawData.daylight_whitebalance)

print("num_colors")
print(rawData.num_colors)

print("raw_colors")
print(rawData.raw_colors)

print("raw_pattern")
print(rawData.raw_pattern)

print("raw_type")
print(rawData.raw_type)

print("rgb_xyz_matrix")
print(rawData.rgb_xyz_matrix)

print("tone_curve")
print(rawData.tone_curve)

print("white_level")
print(rawData.white_level)

print("max()")
print(rawData.raw_image.max())




reference = rawData.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb = True, no_auto_scale = True)
imageio.imsave('./rawpy_noscale.tiff', reference)

'''
reference = rawData.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb = True, no_auto_scale = False)
imageio.imsave('./rawpy_scale.tiff', reference)

reference = rawData.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb = True, no_auto_scale = False)
imageio.imsave('./5219.tiff', reference)
'''

