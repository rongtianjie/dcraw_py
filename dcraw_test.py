import dcraw_utils
import sys
import numpy as np
import imageio
import test_utils

if __name__ == "__main__":

    path = sys.path[0] + "/"
    infn = "DSCF3396.RAF"
    if "." in infn:
        outfn = path + infn.split(".", 1)[0]
    suffix = ".RAF"
    verbose = True
    USE_DARK = True
    dark_frame = ""
    if dark_frame == "":
        USE_DARK = False

    fileList = dcraw_utils.FindAllSuffix(path, suffix, verbose)

    rawData, __ = dcraw_utils.importRawImage(infn, fileList, suffix, verbose)
    # imageio.imsave("raw.tiff", rawData.raw_image)

    rawData_badfix = dcraw_utils.bad_fix(fileList, rawData, verbose)

    if USE_DARK:
        img_sub = dcraw_utils.subtract(rawData_badfix, dark_frame, fileList, verbose)
        scale_in = img_sub
    else:
        scale_in = None

    dcraw_utils.adjust_maximum(rawData_badfix)
    rawImage_wb = dcraw_utils.scale_colors(scale_in, rawData_badfix, verbose)
    # image_demosaiced = demosaicing(rawImage_wb, "RGGB", 2, verbose)
    image_demosaiced = dcraw_utils.demosaicing(rawImage_wb, "RGGB", 0, verbose)

    image_srgb = dcraw_utils.camera_to_srgb(image_demosaiced, rawData_badfix, verbose)

    test_utils.channel_avg(image_srgb)

    imageio.imsave(outfn + "_srgb.tiff", image_srgb.astype(np.uint16))

    reference = rawData.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb = True, no_auto_scale=False)

    test_utils.channel_avg(reference)

    dcraw_utils.save_image_16(outfn + "_ref.tiff", reference, verbose)