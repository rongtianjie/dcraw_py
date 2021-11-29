import cv2
import numpy as np
from collections import OrderedDict
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
import dcraw
import dcraw_utils

def getColorCorrectionSwatches(image_lrgb, IMAGE_BLUR = True):

    if IMAGE_BLUR:
        image_blur = cv2.GaussianBlur(image_lrgb, (9, 9), 0)
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

    print("Found {} swatches.".format(len(SWATCHES)))

    if len(SWATCHES) == 1:
        return SWATCHES[0]
    else:
        print("ERROR!!")

def correction(image_lrgb, swatch):
    # the input image should be 16-bit sRGB
    D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS['ColorChecker 2005']

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        REFERENCE_COLOUR_CHECKER.illuminant, D65,
        colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB)

    swatches_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(swatch, D65, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ))
    colour_checker = colour.characterisation.ColourChecker(
        "src_image", 
        OrderedDict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_xyY)), 
        D65)
    colour.plotting.plot_multi_colour_checkers([REFERENCE_COLOUR_CHECKER, colour_checker])

    swatches_f = colour.colour_correction(swatch, swatch, REFERENCE_SWATCHES)
    swatches_f_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(swatches_f, D65, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ))
    colour_checker = colour.characterisation.ColourChecker(
        '{0} - CC'.format("src_image"),
        OrderedDict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_f_xyY)),
        D65)
    colour.plotting.plot_multi_colour_checkers(
        [REFERENCE_COLOUR_CHECKER, colour_checker])

    cc_image = colour.colour_correction(image_lrgb, swatch, REFERENCE_SWATCHES)
    colour.plotting.plot_image(colour.cctf_encoding(cc_image))

    return cc_image




