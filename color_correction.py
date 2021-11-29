import cv2
import numpy as np
from collections import OrderedDict
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
import dcraw

def srgb_to_lrgb(image_srgb):
    # the input image should be 16-bit sRGB (0-65535)
    image_srgb = image_srgb / 65535
    gamma = ((image_srgb + 0.055) / 1.055) ** 2.4
    scale = image_srgb / 12.92
    image_lrgb = np.where (image_srgb > 0.04045, gamma, scale)
    return dcraw.CLIP((image_lrgb * 65535).astype(np.uint16))

def lrgb_to_srgb(image_lrgb):
    # the input image should be 16-bit linear RGB (0-65535)
    image_lrgb = image_lrgb / 65535
    gamma = 1.055*(image_lrgb ** (1/2.4)) -0.055
    scale = image_lrgb * 12.92
    image_srgb = np.where (image_lrgb > 0.0031308, gamma, scale)
    return dcraw.CLIP((image_srgb * 65535).astype(np.uint16))

def getColorCorrectionSwatches(image_srgb, IMAGE_BLUR = True):

    image_lrgb = srgb_to_lrgb(image_srgb)

    if IMAGE_BLUR:
        image_blur = cv2.GaussianBlur(image_lrgb, (10, 10), 0)
    else:
        image_blur = image_lrgb
    
    swatch = detection(image_blur)

    return swatch

# The input image should be in linear RGB
def detection(image):
    SWATCHES = []
    for swatches, colour_checker, masks in detect_colour_checkers_segmentation(
        image, additional_data=True):
        SWATCHES.append(swatches)

        # Using the additional data to plot the colour checker and masks.
        masks_i = np.zeros(colour_checker.shape)
        for i, mask in enumerate(masks):
            masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1
        colour.plotting.plot_image(
            colour.cctf_encoding(
                np.clip(colour_checker + masks_i * 0.25, 0, 1)))

    if len(SWATCHES) == 1 and len(SWATCHES[0]) == 1:
        return SWATCHES[0][0]

def correction(image, swatch):
    # the input image should be 16-bit sRGB
    image_lrgb = srgb_to_lrgb(image)
    D65 = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    REFERENCE_COLOUR_CHECKER = colour.COLOURCHECKERS['ColorChecker 2005']

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        REFERENCE_COLOUR_CHECKER.illuminant, D65,
        colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB)

    cc_image = colour.colour_correction(image_lrgb, swatch, REFERENCE_SWATCHES, 'Finlayson 2015')

    return lrgb_to_srgb(cc_image)




