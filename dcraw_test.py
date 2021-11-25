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
    dark_frame = ""

    fileList = dcraw_utils.FindAllSuffix(path, suffix, verbose)

    infile = dcraw_utils.findRawImage(infn, fileList, suffix, verbose)

    rawData = dcraw_utils.importRawImage(infile)

    # imageio.imsave("raw.tiff", rawData.raw_image)

    rawData_badfix = dcraw_utils.bad_fix(fileList, rawData, verbose)

    if dark_frame != "":
        img_sub = dcraw_utils.subtract(rawData_badfix, dark_frame, fileList, verbose)
        scale_in = img_sub
    else:
        scale_in = None

    dcraw_utils.adjust_maximum(rawData_badfix, 0.75)
    rawImage_wb = dcraw_utils.scale_colors(scale_in, rawData_badfix, verbose)
    # image_demosaiced = demosaicing(rawImage_wb, "RGGB", 2, verbose)
    image_demosaiced = dcraw_utils.demosaicing(rawImage_wb, "RGGB", 0, verbose)

    image_srgb = dcraw_utils.camera_to_srgb(image_demosaiced, rawData_badfix, verbose)

    dcraw_utils.save_image_16(outfn + "_srgb.tiff", image_srgb, verbose)

    reference = rawData.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb = True, no_auto_scale=False)

    dcraw_utils.save_image_16(outfn + "_demosaic.tiff", image_demosaiced, verbose)

    dcraw_utils.save_image_16(outfn + "_ref.tiff", reference, verbose)