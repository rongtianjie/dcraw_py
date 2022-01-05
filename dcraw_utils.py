
import numpy as np
import rawpy
import rawpy.enhance
import colour_demosaicing
import random
import image_utils


# Define margin for raw data
# Actural effiective resolution is 11664 x 8749
top_margin = 1
bottom_margin = 4
left_margin = 0
right_margin = 144

# Define default pixel max value
maximum = 65535

# sRGB to XYZ matrix
# From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
xyz_srgb = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

# From IEC 61966-2-1:1999
# xyz_srgb = colour.models.RGB_COLOURSPACE_sRGB.RGB_to_XYZ_matrix
d65_white = np.array([0.95047, 1, 1.08883])

# Crop information
# Used to crop the image from libraw size to official size

# GFX100S official image size is 11648 x 8736 (libraw: 11662 x 8752)
class Crop_DB:
    def __init__(self, top_margin, bottom_margin, left_margin, right_margin):
        self.top = top_margin
        self.bottom = bottom_margin
        self.left = left_margin
        self.right = right_margin
crop_GFX100S = Crop_DB(5, 11, 8, 6)

APPLY_BLC = True

def CLIP(src):
    rslt = src.copy()
    rslt[rslt>65536] = 65535
    rslt[rslt<0] = 0
    return rslt

def importRawImage(infPath):
    return rawpy.imread(infPath)

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

def adjust_maximum(raw, maximum_thr):
    global maximum
    real_max = raw.raw_image_visible.max()
    if real_max > 0 and real_max < maximum and real_max > maximum * maximum_thr:
        maximum = real_max

def subtract(raw, dark_img, fileList = None, verbose = False):
    # subtract dark frame to remove noise floor
    # Input: bayer pattern image, dark frame filename,
    if verbose:
        print("Subtraction using dark frame...")

    if fileList == None:
        darkData = importRawImage(dark_img)
    else:
        infPath = image_utils.findRawImage(dark_img, fileList, ".RAF", verbose)
        darkData = importRawImage(infPath)

    darkData_badfix = bad_fix([infPath], darkData, verbose)
    noise_floor = darkData_badfix.raw_image_visible.max()

    if verbose:
        print("The noise floor is {}\n".format(noise_floor))

    rslt = raw.raw_image_visible.astype(np.int32) - noise_floor
    return CLIP(rslt)

def blc(raw, use_pip):
    # BLC on raw image pattern
    # Input should be rawpy object

    # Output will be crop by rawpy "_visible" 
    # On GFX100S, image size is 8752 * 11662
    if use_pip:
        rslt = raw.raw_image[2:8754, :11662].astype(np.int32)
        for i, bl in enumerate(raw.black_level_per_channel):
            rslt[raw.raw_colors[2:8754, :11662] == i] -= bl
    else:
        rslt = raw.raw_image_visible.astype(np.int32)
        for i, bl in enumerate(raw.black_level_per_channel):
            rslt[raw.raw_colors_visible == i] -= bl

    return CLIP(rslt)

def scale_colors(src, raw, use_pip, verbose = False):
    if APPLY_BLC:
        if src==None or src.shape != raw.raw_image_visible.shape:
            src_blc = blc(raw, use_pip)
    else:
        if use_pip:
            src_blc = raw.raw_image[2:8754, :11662].astype(np.int32)
        else:
            src_blc = raw.raw_image_visible.astype(np.int32)

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

    if use_pip:
        scale_matrix = np.empty([8752, 11662], dtype=np.float32)
    else:
        scale_matrix = np.empty(raw.raw_colors_visible.shape, dtype=np.float32)

    for i, scale_co in  enumerate(scale_coeff):
        if use_pip:
            scale_matrix[raw.raw_colors[2:8754, :11662] == i] = scale_co
        else:
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

    return rslt.astype(np.uint16)

def cam_rgb_coeff(cam_xyz):
    # cam_xyz is used to convert color space from XYZ to camera RGB
    cam_xyz = cam_xyz[:3][:]
    cam_rgb = np.dot(xyz_srgb, cam_xyz)
    # Normalize cam_rgb
    cam_rgb_norm = (cam_rgb.T / cam_rgb.sum(axis = 1)).T 
    return cam_rgb_norm
    
def camera_to_srgb(src, raw, verbose = False):
    if verbose:
        print("Start camera rgb to srgb conversion...")

    if src.shape[2] != 3:
        print("The input image should be 3-channel.")
        exit(1)
    else:
        rgb_cam = np.linalg.pinv(cam_rgb_coeff(raw.rgb_xyz_matrix))

    # img_srgb = np.zeros_like(src)
    # for i in range(src.shape[0]):
    #     for j in range(src.shape[1]):
    #         img_srgb[i][j] = np.dot(rgb_cam,src[i][j].T).T

    img_srgb = np.dot(src, rgb_cam.T)
    if verbose:
        print("Conversion done.\n")
    return CLIP(img_srgb).astype(np.uint16)

def auto_bright(image_srgb, perc = 0.01, verbose = False):
    # Use percentile to auto bright image
    
    if verbose:
        print("\nStart auto bright...")

    white_num = int(image_srgb.shape[0] * image_srgb.shape[1] * perc)

    gray = image_utils.rgb2gray(image_srgb)
    
    hist = np.bincount(gray.ravel(), minlength=65536)

    cnt = 0
    white = 0
    for i in range(len(hist)):
        cnt += hist[65535 - i]
        if cnt > white_num:
            white = 65535 - i
            break

    ratio = 65535 / white
    if verbose:
        print("Brighten ratio: {}".format(ratio))

    image_bright = image_srgb * ratio

    if verbose:
        print("Auto bright finished.")
    return CLIP(image_bright).astype(np.uint16)

def crop_to_official_size(img, model = "GFX100S", verbose = False):
    if model == "GFX100S":
        rslt = image_utils.crop_image(img, crop_GFX100S.top, crop_GFX100S.bottom, crop_GFX100S.left, crop_GFX100S.right)
        if verbose:
            print("Output image size is {} x {}".format(rslt.shape[1], rslt.shape[0]))
        return rslt

if __name__ == "__main__":
    print("This is the dcraw utils script.")
    exit(0)