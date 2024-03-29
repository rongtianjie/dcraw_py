import cv2
import numpy as np
from collections import OrderedDict
import colour
from colour_checker_detection import detect_colour_checkers_segmentation

class CreateSpyderCheck:
    name = "SpyderChecker 24"
    # Color checker reference values are in xyY color space
    data = OrderedDict()
    data["Aqua"] = np.array([0.29131, 0.39533, 0.4102])
    data["Lavender"] = np.array([0.29860, 0.28411, 0.22334])
    data["Evergreen"] = np.array([0.36528, 0.46063, 0.12519])
    data["Steel Blue"] = np.array([0.27138, 0.29748, 0.17448])
    data["Classic Light Skin"] = np.array([0.42207, 0.37609, 0.34173])
    data["Classic Dark Skin"] = np.array([0.44194, 0.38161, 0.09076])
    data["Primary Orange"] = np.array([0.54238, 0.40556, 0.2918])
    data["Blueprint"] = np.array([0.22769, 0.21517, 0.09976])
    data["Pink"] = np.array([0.50346, 0.32519, 0.1826])
    data["Violet"] = np.array([0.30813, 0.24004, 0.05791])
    data["Apple Green"] = np.array([0.40262, 0.50567, 0.44332])
    data["Sunflower"] = np.array([0.50890, 0.43959, 0.4314])
    data["Primary Cyan"] = np.array([0.19792, 0.30072, 0.16111])
    data["Primary Magenta"] = np.array([0.38429, 0.23929, 0.18286])
    data["Primary Yellow"] = np.array([0.47315, 0.47936, 0.63319])
    data["Primary Red"] = np.array([0.59685, 0.31919, 0.11896])
    data["Primary Green"] = np.array([0.32471, 0.51999, 0.22107])
    data["Primary Blue"] = np.array([0.19215, 0.15888, 0.04335])
    data["Card White"] = np.array([0.35284, 0.36107, 0.90104])
    data["20% Gray"] = np.array([0.35137, 0.36134, 0.57464])
    data["40% Gray"] = np.array([0.35106, 0.36195, 0.34707])
    data["60% Gray"] = np.array([0.35129, 0.36209, 0.18102])
    data["80% Gray"] = np.array([0.35181, 0.36307, 0.07794])
    data["Card Black"] = np.array([0.34808, 0.35030, 0.02284])

    # CIE D65 XYZ
    # From https://cdn-10.nikon-cdn.com/pdf/NS4_man.pdf
    # illuminant = np.array([0.3127159, 0.3290015])

    # CIE D50 XYZ
    illuminant = np.array([ 0.34570291,  0.3585386 ])

def auto_bright(image_lrgb, dst_avg = 0.17, verbose = False):
    ratio = 1
    if image_lrgb.mean() < dst_avg:
        ratio = dst_avg / image_lrgb.mean()
        image_lrgb *= ratio
        if verbose:
            print("Auto bright ratio: {}".format(ratio))
    return image_lrgb, ratio

def getColorCorrectionSwatches(image_lrgb, auto_shink = False, IMAGE_BLUR = 11, verbose = False):
    # The input image should convert to linear RGB with colour.cctf_decoding()
    if auto_shink:
        if max(image_lrgb.shape[0], image_lrgb.shape[1]) > 1500:
            ratio = 1000 / max(image_lrgb.shape[0], image_lrgb.shape[1])
            image_lrgb = cv2.resize(image_lrgb, (0, 0), fx = ratio, fy = ratio)
    if IMAGE_BLUR:
        image_blur = cv2.GaussianBlur(image_lrgb, (IMAGE_BLUR, IMAGE_BLUR), 0)
    else:
        image_blur = image_lrgb
    
    swatch = detection(image_blur, 48, verbose)

    return swatch

# The input image should be in linear RGB
def detection(image, samples_size, verbose = False):
    SWATCHES = []

    for colour_checker_swatches_data in detect_colour_checkers_segmentation(
        image, samples=samples_size, additional_data=True):
        swatch_colours, colour_checker_image, swatch_masks = (
            colour_checker_swatches_data.values)
        SWATCHES.append(swatch_colours)
        
        if verbose:
            # Using the additional data to plot the colour checker and masks.
            masks_i = np.zeros(colour_checker_image.shape)
            for i, mask in enumerate(swatch_masks):
                masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1
            colour.plotting.plot_image(
                colour.cctf_encoding(
                    np.clip(colour_checker_image + masks_i * 0.25, 0, 1)))
    if verbose:
        print("Found {} swatches.".format(len(SWATCHES)))

    if len(SWATCHES) == 1:
        return SWATCHES[0]
    else:
        if verbose:
            print("ERROR. Can't find or found multiple swatches.")
        return None

def correction(image_lrgb, swatch, checker = CreateSpyderCheck(), verbose = False):
    # the input image should be 16-bit sRGB
    D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    REFERENCE_COLOUR_CHECKER = checker
    # REFERENCE_COLOUR_CHECKER = colour.COLOURCHECKERS['ColorChecker 2005']

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        REFERENCE_COLOUR_CHECKER.illuminant, D65,
        colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB)
    if verbose:
        # print(REFERENCE_SWATCHES)
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
    
    # if np.min(cc_image) < 0:
    #     cc_image -= np.min(cc_image)
    # cc_image = cc_image/np.max(cc_image)

    if verbose:
        colour.plotting.plot_image(colour.cctf_encoding(cc_image))

    return cc_image


if __name__ == "__main__":
    print("This is the colour correction w/ colour checker script.")
    exit(0)


