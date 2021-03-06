from dcraw_utils import *
from other.image_utils import *
import sys
import getopt
import os

def imread(infile, path = None, suffix = ".RAF", verbose = False):
    if path == None:
        rawData = importRawImage(infile)
    else:
        fileList = findAllSuffix(path, suffix, verbose)
        infPath = findRawImage(infile, fileList, suffix, verbose)
        rawData = importRawImage(infPath)
    return rawData

def postprocessing(rawData, use_rawpy_postprocessing = False, suffix = ".RAF", path = None, bad_pixel_fix = True, demosacing_method = 0, output_srgb = False, auto_bright = False, bright_perc = None, crop_to_official = False, cam_model = None, verbose = False):
    
    debug = False

    if use_rawpy_postprocessing:
        return rawData.postprocess(gamma = (1, 1), no_auto_bright = True, output_bps = 16, use_camera_wb = True)

    if path == None or bad_pixel_fix == False:
        rawData_badfix = rawData
    else:
        fileList = findAllSuffix(path, suffix, verbose)
        rawData_badfix = bad_fix(rawData, fileList, verbose)

    rawData = overwrite_imagedata(rawData_badfix, cam_model, verbose)

    rawImage_blc = blc(rawData)

    rawImage_ge = green_channel_equilibrium(rawImage_blc, rawData)

    rawImage_wb = scale_colors(rawImage_ge, rawData, verbose)
    
    image_demosaiced = demosaicing(rawImage_wb, rawData, demosacing_method, verbose)

    if debug:
        save_image_16("debug_demosaiced.tiff", image_demosaiced)

    if output_srgb:
        output = camera_to_srgb(image_demosaiced, rawData.rgb_xyz_matrix, verbose)
        if debug:
            save_image_16("debug_srgb.tiff", output)
    else:
        output = image_demosaiced

    if auto_bright:
        output = auto_bright(output, bright_perc, rawData.white_level, verbose)
        if debug:
            save_image_16("autobright_srgb.tiff", output)
    
    if crop_to_official:
        output = crop_to_official_size(output, verbose)

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

    rgb = postprocessing(rawData, cam_model = "GFX100S", verbose = verbose)

    save_image_16(outfn + ".tiff", rgb, verbose = verbose)

    exit(0)