import dcraw_utils
import sys
import getopt
import os
from other.image_utils import findAllSuffix, findRawImage, save_image_16

def imread(infile, path = None, suffix = ".RAF", verbose = False):
    if path == None:
        rawData = dcraw_utils.importRawImage(infile)
    else:
        fileList = findAllSuffix(path, suffix, verbose)
        infPath = findRawImage(infile, fileList, suffix, verbose)
        rawData = dcraw_utils.importRawImage(infPath)
    return rawData

def postprocessing(rawData, use_rawpy_postprocessing = False, suffix = ".RAF", adjust_maximum_thr = 0.75, dark_frame = None, path = None, bad_pixel_fix = True, bayer_pattern = "RGGB", demosacing_method = 0, output_srgb = False, auto_bright = False, bright_perc = 0.01, crop_to_official = False, use_pip = False, verbose = False):
    
    debug = False

    if use_rawpy_postprocessing:
        return rawData.postprocess(gamma = (1, 1), no_auto_bright = True, output_bps = 16, use_camera_wb = True)

    if path == None or bad_pixel_fix == False:
        rawData_badfix = rawData
    else:
        fileList = findAllSuffix(path, suffix, verbose)
        rawData_badfix = dcraw_utils.bad_fix(fileList, rawData, verbose)
    
    # dcraw_utils.adjust_maximum(rawData_badfix, adjust_maximum_thr)

    if dark_frame == None:
        rawImage_wb = dcraw_utils.scale_colors(None, rawData_badfix, use_pip, verbose)
    else:
        img_sub = dcraw_utils.subtract(rawData_badfix, dark_frame, fileList, verbose)
        rawImage_wb = dcraw_utils.scale_colors(img_sub, rawData_badfix, verbose)
    
    image_demosaiced = dcraw_utils.demosaicing(rawImage_wb, bayer_pattern, demosacing_method, verbose)

    if debug:
        save_image_16("debug_demosaiced.tiff", image_demosaiced)

    if output_srgb:
        output = dcraw_utils.camera_to_srgb(image_demosaiced, rawData_badfix, verbose)
        if debug:
            save_image_16("debug_srgb.tiff", output)
    else:
        output = image_demosaiced

    if auto_bright:
        output = dcraw_utils.auto_bright(output, bright_perc, verbose)
    
    if crop_to_official:
        output = dcraw_utils.crop_to_official_size(output, verbose = verbose)

    print("Dcraw finished.")
    
    return output


        

if __name__ == "__main__":

    path = sys.path[0] + "/"
    # infn = ""
    infn = "DSCF9329.RAF"
    if "." in infn:
        outfn = path + infn.split(".", 1)[0]
    suffix = ".RAF"
    verbose = True

    # This part is out-dated
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

    rawData = imread(infn, path = path, verbose = verbose)

    rgb = postprocessing(rawData, use_pip=True, verbose = verbose)

    save_image_16(outfn + ".tiff", rgb, verbose = verbose)

    exit(0)