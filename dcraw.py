import dcraw_utils
import sys
import getopt
import os

def imread(infile, path = None, suffix = ".RAF", verbose = False):
    if path == None:
        rawData = dcraw_utils.importRawImage(infile)
    else:
        fileList = dcraw_utils.FindAllSuffix(path, suffix, verbose)
        infPath = dcraw_utils.findRawImage(infn, fileList, suffix, verbose)
        rawData = dcraw_utils.importRawImage(infPath)
    return rawData

def postprocessing(rawData, suffix = ".RAF", adjust_maximum_thr = 0.75, dark_frame = None, path = None, bayer_pattern = "RGGB", demosacing_method = 0, verbose = False):
    if path == None:
        rawData_badfix = rawData
    else:
        fileList = dcraw_utils.FindAllSuffix(path, suffix, verbose)
        rawData_badfix = dcraw_utils.bad_fix(fileList, rawData, verbose)
    dcraw_utils.adjust_maximum(rawData_badfix, adjust_maximum_thr)

    if dark_frame == None:
        rawImage_wb = dcraw_utils.scale_colors(None, rawData_badfix, verbose)
    else:
        img_sub = dcraw_utils.subtract(rawData_badfix, dark_frame, fileList, verbose)
        rawImage_wb = dcraw_utils.scale_colors(img_sub, rawData_badfix, verbose)
    
    image_demosaiced = dcraw_utils.demosaicing(rawImage_wb, bayer_pattern, demosacing_method, verbose)

    image_srgb = dcraw_utils.camera_to_srgb(image_demosaiced, rawData_badfix, verbose)

    return image_srgb


if __name__ == "__main__":

    path = sys.path[0] + "/"
    # infn = ""
    infn = "DSCF3396.RAF"
    if "." in infn:
        outfn = path + infn.split(".", 1)[0]
    suffix = ".RAF"
    verbose = True

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

    img_rgb = postprocessing(rawData, verbose = verbose)

    dcraw_utils.save_image_16(outfn + "_srgb.tiff", img_rgb, verbose = verbose)