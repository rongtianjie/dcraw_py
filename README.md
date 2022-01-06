# Dcraw implemented by Python (continuously updating...)

## Introduction

 A self modified raw image decoder using rawpy to unpack the raw file.

 For now, the script support FujiFilm GFX100S

## Usage

 Simply import the `dcraw` library in your Python script.

 ```
 rawData = dcraw.imread(infile, path = None, suffix = ".RAF", verbose = False)
 ``` 
 will return a `rawpy.RawPy` object. Almost the same as `rawpy.imread()`.

The main ISP implementation is contained in 
 ```
 output = dcraw.postprocessing(rawData, use_rawpy_postprocessing = False, suffix = ".RAF", path = None, bad_pixel_fix = True, demosacing_method = 0, output_srgb = False, auto_bright = False, bright_perc = None, crop_to_official = False, cam_model = None, verbose = False)
 ```
 For a simple usage for FujiFIlm GFX100S:
 ```
 output = dcraw.postprocessing(rawData, cam_model = "GFX100S")
 ```

#### Parameters
- **infile** (str) - The input image filename. 

    - if `path` is defined, the `infile` do not have to containe the whole file path. The script will automatically search all the files under `path` matches the `path` including the subfolders. If there are multiple files match the keyword, the script will launch a choosen selection prompt.

    - if `path` is not defined (default None), the `infile` should be the absolute path or relative path to the script.

- **path** (str) - The root path of all the image files. This parameter has 2 functions:

    - It can be used in the searching for the absolute file path, including the input image and dark frame.

    - The images used for bad pixel removal will be randomly selected from this folder.

- **suffix** (str) - The suffix for the image file. ".RAF" for Fujifilm cameras.

- **use_rawpy_postprocessing** (bool) - Whether use rawpy built-in postprocessing.

    - Linear gamma output

    - Do not auto bright image

    - Output 16-bit image

    - Use camera white balance setting

- **bad_pixel_fix** (bool) - Whether apply the bad pixel fix module. Requiring determining the `path` parameter.

- **demosacing_method** (int) - Default: 0 - bilinear
    - 0 - colour_demosaicing.demosaicing_CFA_Bayer_bilinear
    - 1 - colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004
    - 2 - colour_demosaicing.demosaicing_CFA_Bayer_Menon2007

        > Malvar2004 will generate false color when image has bright dots.
        >
        > Menon2007 needs more than 20 GB memory.

- **output_srgb** (bool) - Whether output srgb color space. Default: False

- **auto_bright** (bool) - Whether bright the image automatically. Default: False

- **bright_perc** (float) - If use auto bright, a brightest part of pixel will be set to pure white. The ratio is controled by `bright_prec` (Default: None, Recommend: 0.01)

- **crop_to_official** (bool) - Whether crop the output into Camera official size (Default: False)

- **cam_model** (str) - Whether use the pre-defined raw image parameters. (Default: None)

    **Current support cameras**:

    - FujiFilm GFX100S ("GFX100S")

- **verbose** (bool) - Whether showing the progress log.

## Reference

[Libraw docs](https://www.libraw.org/docs)

[rawpy api reference](https://letmaik.github.io/rawpy/api/index.html)

[dcraw annotated & outlined](https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html)



