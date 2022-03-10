import numpy as np
from scipy.ndimage.filters import convolve
from colour.utilities import tstack

def mhc_demosaic(cfa, raw):

    Gr_Gb = np.asarray([[0, 0, -1, 0, 0], [0, 0, 2, 0 , 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype=np.float64) / 8

    Rg_r_Bg_r = np.asarray([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], dtype=np.float64) / 8

    Rg_b_Bg_b = np.transpose(Rg_r_Bg_r)

    Rb_Br = np.asarray([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], dtype=np.float64) / 8

    R = np.zeros(cfa.shape, dtype=np.float64)
    G = np.zeros(cfa.shape, dtype=np.float64)
    B = np.zeros(cfa.shape, dtype=np.float64)

    # R pixels
    R[raw.raw_colors_visible==0] = cfa[raw.raw_colors_visible==0]
    G[raw.raw_colors_visible==0] = convolve(cfa, Gr_Gb)[raw.raw_colors_visible==0]
    B[raw.raw_colors_visible==0] = convolve(cfa, Rb_Br)[raw.raw_colors_visible==0]

    # G pixels at R rows
    R[raw.raw_colors_visible==1] = convolve(cfa, Rg_r_Bg_r)[raw.raw_colors_visible==1]
    G[raw.raw_colors_visible==1] = cfa[raw.raw_colors_visible==1]
    B[raw.raw_colors_visible==1] = convolve(cfa, Rg_r_Bg_r)[raw.raw_colors_visible==1]

    # B pixels
    R[raw.raw_colors_visible==2] = convolve(cfa, Rb_Br)[raw.raw_colors_visible==2]
    B[raw.raw_colors_visible==2] = cfa[raw.raw_colors_visible==2]
    G[raw.raw_colors_visible==2] = convolve(cfa, Gr_Gb)[raw.raw_colors_visible==2]
    
    # G pixels at B rows
    R[raw.raw_colors_visible==3] = convolve(cfa, Rg_b_Bg_b)[raw.raw_colors_visible==3]
    G[raw.raw_colors_visible==3] = cfa[raw.raw_colors_visible==3]
    B[raw.raw_colors_visible==3] = convolve(cfa, Rg_b_Bg_b)[raw.raw_colors_visible==3]

    # Combine 3 channels
    return tstack([R, G, B])

if __name__ == '__main__':
    pass