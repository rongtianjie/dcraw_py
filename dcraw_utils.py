import numpy as np
import rawpy
import rawpy.enhance
import cam_data
import colour_demosaicing
import random
from other.image_utils import findAllSuffix, findRawImage, rgb2gray, crop_image
from demosaic_pack import *

# Define margin for raw data
# Actural effiective resolution is 11664 x 8749
raw_top = 1
raw_bottom = 4
raw_left = 0
raw_right = 144

# sRGB to XYZ matrix
# From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
xyz_srgb = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

# From IEC 61966-2-1:1999
# xyz_srgb = colour.models.RGB_COLOURSPACE_sRGB.RGB_to_XYZ_matrix
d65_white = np.array([0.95047, 1, 1.08883])


class rawData:
    raw_image_visible = None
    raw_colors_visible = None
    def __init__(self, blpc, wlpc, cwb, cd, cm, dwb, nc, rc, ri, rp, rt, rgbxyz, sizes, tc, wl):
        self.black_level_per_channel = blpc
        self.camera_white_level_per_channel = wlpc
        self.camera_whitebalance = cwb
        self.color_desc = cd
        self.color_matrix = cm
        self.daylight_whitebalance = dwb
        self.num_colors = nc
        self.raw_colors = rc
        self.raw_image = ri
        self.raw_pattern = rp
        self.raw_type = rt
        self.rgb_xyz_matrix = rgbxyz
        self.sizes = sizes
        self.tone_curve = tc
        self.white_level = wl

def CLIP(src):
    rslt = src.copy()
    rslt[rslt>65536] = 65535
    rslt[rslt<0] = 0
    return rslt

def importRawImage(infPath):
    return rawpy.imread(infPath)

def bad_fix(rawData, fileList, verbose):
    # Fix bad pixels with rawpy.enhance

    # Input the rawpy object and image list
    # return a rawpy object with bad pixel fixed 
    if len(fileList) >= 1:

        if verbose:
            print("Starting bad pixel fixing.")

        if len(fileList) <= 10:
            sample_num = len(fileList)
        else:
            sample_num = min(10, len(fileList)//2)

        blc_sample = random.sample(fileList, sample_num)

        if verbose:
            print("Finding bad pixel using {} image(s)...".format(sample_num))
            
        bad_list = rawpy.enhance.find_bad_pixels(blc_sample, find_hot=True, find_dead=False, confirm_ratio=0.9)
        
        if verbose:
            print("Found {} bad pixels:".format(len(bad_list)))
        
        rawpy.enhance.repair_bad_pixels(rawData, bad_list, method='median')
        
        if verbose:
            print("Bad pixel fixing finished.\n")

    return rawData

def overwrite_imagedata(r, cam_model, verbose):

    if cam_model == "GFX100S":
        if verbose:
            print("Overwrite image data with GFX100S.")
        data = cam_data.GFX100S()
        raw = rawData(data.black_level_per_channel, r.camera_white_level_per_channel, r.camera_whitebalance, data.color_desc, r.color_matrix, data.daylight_whitebalance, data.num_colors, r.raw_colors, r.raw_image, data.raw_pattern, r. raw_type, data.rgb_xyz_matrix, data.sizes, data.tone_curve, data.white_level)

        raw.raw_image_visible = raw.raw_image[raw.sizes.top_margin:raw.sizes.top_margin+raw.sizes.iheight, raw.sizes.left_margin:raw.sizes.left_margin+raw.sizes.iwidth]
        raw.raw_colors_visible = raw.raw_colors[raw.sizes.top_margin:raw.sizes.top_margin+raw.sizes.iheight, raw.sizes.left_margin:raw.sizes.left_margin+raw.sizes.iwidth]
    else:
        if verbose:
            print("Overwrite image data with raw file.")
        raw = rawData(r.black_level_per_channel, r.camera_white_level_per_channel, r.camera_whitebalance, r.color_desc, r.color_matrix, r.daylight_whitebalance, r.num_colors, r.raw_colors, r.raw_image, r.raw_pattern, r. raw_type, r.rgb_xyz_matrix, r.sizes, r.tone_curve, r.white_level)

        raw.raw_image_visible = r.raw_image_visible
        raw.raw_colors_visible = r.raw_colors_visible
    
    return raw

def blc(raw):
    # BLC on raw image pattern
    # Input should be rawpy object

    # Output will be crop by rawpy "_visible" 
    # On GFX100S, image size is 8752 * 11662

    rslt = raw.raw_image_visible.astype(np.int32)
    for i, bl in enumerate(raw.black_level_per_channel):
        rslt[raw.raw_colors_visible == i] -= bl

    return CLIP(rslt)

def green_channel_equilibrium(src, raw):
    g2_ratio = src[raw.raw_colors_visible == 1].sum() / src[raw.raw_colors_visible == 3].sum()
    m = np.ones(src.shape, dtype=np.float32)
    m[raw.raw_colors_visible == 3] = g2_ratio
    rslt = src * m
    return CLIP(rslt).astype(np.uint16)

def scale_colors(src, raw, verbose):
    if verbose:
        print("Start white balance correction with camera setting.")

    wb = raw.camera_whitebalance
    wb_coeff = np.asarray(wb[:3]) / max(wb[:3])
    wb_coeff = np.append(wb_coeff,wb_coeff[1])

    if verbose:
        print("WB coefficient is {}".format(wb_coeff))
    
    if raw.camera_white_level_per_channel is None:
        white_level = [65535] * 4
    else:
        white_level = raw.camera_white_level_per_channel
    
    white_level = np.array(white_level) - np.array(raw.black_level_per_channel)

    scale_coeff = wb_coeff * 65535 / white_level
    if verbose:
        print("Scale coefficient is {}".format(scale_coeff))

    scale_matrix = np.empty([raw.sizes.iheight, raw.sizes.iwidth], dtype=np.float32)

    for i, scale_co in  enumerate(scale_coeff):
        scale_matrix[raw.raw_colors_visible == i] = scale_co

    rslt = CLIP(src * scale_matrix)

    if verbose:
        print("White balance finished.\n")

    return rslt.astype(np.uint16)

def demosaicing(src, raw, DEMOSACING_METHOD, verbose):
    # Demosaicing. Default using colour_demosaicing.demosaicing_CFA_Bayer_bilinear
    if verbose:
        print("Start demosaicing.")
    color_desc = str(raw.color_desc, 'utf-8')

    Bayer_Pattern = color_desc[:2] + color_desc[3] + color_desc[2]

    if DEMOSACING_METHOD < 3:
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

    elif DEMOSACING_METHOD == 4:
        if verbose:
            print("Demosacing using AMaZE")
        rslt = amaze_demosaic(src, raw)

    else:
        print("Can not find the DEMOSACING_METHOD.")
    
    if verbose:
        print("Demosacing finished.\n")

    return CLIP(rslt).astype(np.uint16)

def cam_rgb_coeff(cam_xyz):
    # cam_xyz is used to convert color space from XYZ to camera RGB
    cam_xyz = cam_xyz[:3][:]
    cam_rgb = np.dot(xyz_srgb, cam_xyz)
    # Normalize cam_rgb
    cam_rgb_norm = (cam_rgb.T / cam_rgb.sum(axis = 1)).T 
    return cam_rgb_norm

def camera_to_srgb(src, rgb_xyz_matrix, verbose):
    if verbose:
        print("Start camera rgb to srgb conversion...")

    if src.shape[2] != 3:
        print("The input image should be 3-channel.")
        exit(1)
    else:
        rgb_cam = np.linalg.pinv(cam_rgb_coeff(rgb_xyz_matrix))

    # img_srgb = np.zeros_like(src)
    # for i in range(src.shape[0]):
    #     for j in range(src.shape[1]):
    #         img_srgb[i][j] = np.dot(rgb_cam,src[i][j].T).T

    img_srgb = np.dot(src, rgb_cam.T)
    if verbose:
        print("Conversion done.\n")
    return CLIP(img_srgb).astype(np.uint16)

def auto_bright(image_srgb, perc, white_level, verbose):
    # Use percentile to auto bright image
    if perc is not None:
        if verbose:
            print("\nStart auto bright...")

        white_num = int(image_srgb.shape[0] * image_srgb.shape[1] * perc)

        gray = rgb2gray(image_srgb)
        
        hist = np.bincount(gray.ravel(), minlength=65536)

        cnt = 0
        white = 0
        for i in range(len(hist)):
            cnt += hist[65535 - i]
            if cnt > white_num:
                white = 65535 - i
                break
    elif white_level < 65535:
        white = white_level

    ratio = 65535 / white
    if verbose:
        print("Brighten ratio: {}".format(ratio))

    image_bright = image_srgb * ratio

    if verbose:
        print("Auto bright finished.")
    return CLIP(image_bright).astype(np.uint16)

def crop_to_official_size(img, cam_model, verbose):
    if cam_model == "GFX100S":
        crop_GFX100S = cam_data.CROP_GFX100S()
        rslt = crop_image(img, crop_GFX100S.top, crop_GFX100S.bottom, crop_GFX100S.left, crop_GFX100S.right)
        if verbose:
            print("Output image size is {} x {}".format(rslt.shape[1], rslt.shape[0]))
        return rslt

if __name__ == "__main__":
    print("This is the dcraw utils script.")
    exit(0)