import dcraw_utils
import sys
import numpy as np
import imageio
import test_utils

if __name__ == "__main__":

    path = sys.path[0]
    infn = "DSCF3396.RAF"
    outfn = "demosaiced.tiff"
    suffix = ".RAF"
    verbose = True

    fileList = dcraw_utils.FindAllSuffix(path, suffix, verbose)

    rawData = dcraw_utils.importRawImage(infn, fileList, suffix, verbose)
    # imageio.imsave("raw.tiff", rawData.raw_image)

    rawData_badfix = dcraw_utils.bad_fix(fileList, rawData, verbose)

    dcraw_utils.adjust_maximum(rawData_badfix)
    
    rawImage_wb = dcraw_utils.scale_colors(rawData_badfix, verbose)

    # image_demosaiced = demosaicing(rawImage_wb, "RGGB", 2, verbose)
    image_demosaiced = dcraw_utils.demosaicing(rawData_badfix.raw_image_visible, "RGGB", 0, verbose)

    test_utils.channe_avg(image_demosaiced)

    imageio.imsave("demosaic.tiff", image_demosaiced.astype(np.uint16))

    reference = rawData.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb = True, no_auto_scale=True)

    test_utils.channe_avg(reference)

    imageio.imsave("ref.tiff", reference.astype(np.uint16))