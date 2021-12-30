# Dcraw implemented by Python (continuously updating...)

## Introduction

 A self modified raw image decode work flow based on rawpy

 For now, the script is designed for FujiFilm GFX100S

## Usage

 Simply import the `dcraw` library in your Python script.

 ```
 rawData = dcraw.imread(infile, path = None, suffix = ".RAF", verbose = False)
 ``` 
 will return a `rawpy.RawPy` object. Almost the same as `rawpy.imread()`.

The main ISP implementation is contained in 
 ```
 output = dcraw.postprocessing(rawData, use_rawpy_postprocessing = False, suffix = ".RAF", adjust_maximum_thr = 0.75, dark_frame = None, path = None, bad_pixel_fix = True, bayer_pattern = "RGGB", demosacing_method = 0, output_srgb = False, auto_bright = False, bright_perc = 0.01, crop_to_official = False, use_pip = False, verbose = False)
 ```

#### Parameters
- **infile** (str) - The input image filename. 

    - if `path` is defined, the `infile` do not have to containe the whole file path. The script will automatically search all the files under `path` matches the `path` including the subfolders. If there are multiple files match the keyword, the script will launch a choosen selection prompt.

    - if `path` is not defined (default None), the `infile` should be the absolute path or relative path to the script.

- **use_rawpy_postprocessing** (bool) - Whether use rawpy built-in postprocessing.

    - Linear gamma output

    - Do not auto bright image

    - Output 16-bit image

    - Use camera white balance setting

- **path** (str) - The root path of all the image files. This parameter has 2 functions:

    - It can be used in the searching for the absolute file path, including the input image and dark frame.

    - The images used for bad pixel removal will be randomly selected from this folder.

- **suffix** (str) - The suffix for the image file. ".RAF" for Fujifilm cameras.

- **adjust_maximum_thr** (float) - See libraw docs. [Here](https://www.libraw.org/docs/API-datastruct-eng.html#libraw_output_params_t)

- **dark_frame** (str) - The dark frame filename. Used to remove the noise floor. Similar to `infile`, file path also can be auto generated with path defined.

- **bad_pixel_fix** (bool) - Whether apply the bad pixel fix module. Requiring determining the `path` parameter.

- **bayer_pattern** (str) - The bayer pattern of camera cmos.
    > The original cmos bayer pattern of Fujifilm GFX100S is "GBRG". However, it turns to "RGGB" after the raw image data is cropped by "_visible".

- **demosacing_method** (int) - Default: 0 - bilinear
    - 0 - colour_demosaicing.demosaicing_CFA_Bayer_bilinear
    - 1 - colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004
    - 2 - colour_demosaicing.demosaicing_CFA_Bayer_Menon2007

        > Malvar2004 will generate false color when image has bright dots.
        >
        > Menon2007 needs more than 20 GB memory.

- **output_srgb** (bool) - Whether output srgb color space. Default: False
    - If set to "True", the functurn will return 2 values [linear, srgb]

- **auto_bright** (bool) - Whether bright the image automatically. Default: False

- **bright_perc** (float) - If use auto bright, a brightest part of pixel will be set to pure white. The ratio is controled by `bright_prec` (Default: 0.01)

- **crop_to_official** (bool) - Whether crop the output into Fujifilm official size (Default: False)

- **use_pip** (bool) - When set to True, can directly use `pip` to install `rawpy`

- **verbose** (bool) - Whether showing the progress log.

## Reference

[Libraw docs](https://www.libraw.org/docs)

[rawpy api reference](https://letmaik.github.io/rawpy/api/index.html)

[dcraw annotated & outlined](https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html)



