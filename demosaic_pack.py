import numpy as np
from numba import jit

def fc(cfa, r, c):
    return cfa[r&1, c&1]


def intp(a, b, c):
    return a * (b - c) + c

def SQR(x):
    return x ** 2

def amaze_demosaic(src, raw):

    cfarray = raw.raw_colors
    cfarray[cfarray == 3] = 1

    rgb = amaze_demosaic_libraw(src, cfarray, raw.daylight_whitebalance)

    return rgb

def amaze_demosaic_darktable(src, cfarray, daylight_wb):

    TS = 512
    TSh = int(TS/2)
    winx = winy = 0
    width = src.shape[1]
    height = src.shape[0]
    clip_pt = min(daylight_wb[:3])

    if fc(cfarray, 0, 0)== 1:
        if fc(cfarray, 0, 1) == 0:
            ex, ey = 1, 0
        else:
            ex, ey = 0, 1
    else:
        if fc(cfarray, 0, 0) == 0:
            ex = ey = 0
        else: 
            ex = ey = 1

    v1 = TS
    v2 = 2 * TS
    v3 = 3 * TS
    p1 = -TS + 1
    p2 = -2 * TS + 2
    p3 = -3 * TS + 3
    m1 = TS + 1 
    m2 = 2 * TS + 2
    m3 = 3 * TS + 3

    eps, epssq = 1e-5, 1e-10
    arthresh = 0.75

    gaussodd = [0.14659727707323927, 0.103592713382435, 0.0732036125103057, 0.0365543548389495]
    nyqthresh = 0.5
    gaussgrad = [nyqthresh * 0.07384411893421103, nyqthresh * 0.06207511968171489, nyqthresh * 0.0521818194747806, nyqthresh * 0.03687419286733595, nyqthresh * 0.03099732204057846, nyqthresh * 0.018413194161458882]
    gausseven = [0.13719494435797422, 0.05640252782101291]
    gquinc = [0.169917, 0.108947, 0.069855, 0.0287182]
    
    loop_cnt = 1

    for top in range(winy-16, winy+height, TS-32):
        for left in range(winx-16, winx+width, TS-32):
            print("Loop [{}]: top: {} left: {}".format(loop_cnt, top, left))
            loop_cnt += 1
            bottom = min(top+TS, winy+height+16)
            right = min(left+TS, winx+width+16)
            rr1 = bottom - top
            cc1 = right - left
            
            rrmin = 16 if top < winy else 0
            ccmin = 16 if left < winx else 0
            rrmax = winy+height-top if bottom>(winy+height) else rr1
            ccmax = winx+width-left if right>(winx+width) else cc1

            rgbgreen = np.empty(TS*TS, dtype=np.float32)
            cfa = np.empty(TS*TS, dtype=np.float32)
            delhvsqsum = np.empty(TS*TS, dtype=np.float32)
            dirwTS0 = np.empty(TS*TS, dtype=np.float32)
            dirwTS1 = np.empty(TS*TS, dtype=np.float32)
            vcd = np.empty(TS*TS, dtype=np.float32)
            hcd = np.empty(TS*TS, dtype=np.float32)
            vcdalt = np.empty(TS*TS, dtype=np.float32)
            hcdalt = np.empty(TS*TS, dtype=np.float32)
            dgintv = np.empty(TS*TS, dtype=np.float32)
            dginth = np.empty(TS*TS, dtype=np.float32)
            cddiffsq = np.empty(TS*TS, dtype=np.float32)
            hvwt = np.empty(TS*TS, dtype=np.float32)
            nyquist = np.empty(TS*TS, dtype=np.float32)
            nyqutest = np.empty(TS*TS, dtype=np.float32)
            Dgrb = np.empty([2, TS*TS], dtype=np.float32)
            Dgrbh2 = np.empty(TS*TS, dtype=np.float32)
            Dgrbv2 = np.empty(TS*TS, dtype=np.float32)
            delp = np.empty(TS*TS, dtype=np.float32)
            delm = np.empty(TS*TS, dtype=np.float32)
            Dgrbsq1p = np.empty(TS*TS, dtype=np.float32)
            Dgrbsq1m = np.empty(TS*TS, dtype=np.float32)
            rbm = np.empty(TS*TS, dtype=np.float32)
            rbp = np.empty(TS*TS, dtype=np.float32)
            pmwt = np.empty(TS*TS, dtype=np.float32)
            rbint = np.empty(TS*TS, dtype=np.float32)
            RGB = np.empty([height, width, 3], dtype=np.uint16)
            
            
            # fill upper border
            if rrmin > 0:
                for rr in range(16):
                    for cc in range(ccmin, ccmax):
                        row = 32 - rr + top
                        cfa[rr*TS+cc] = src[row, cc+left] / 65535
                        rgbgreen[rr*TS+cc] = cfa[rr*TS+cc]

            # fill inner part
            for rr in range(rrmin, rrmax):
                row = rr + top
                for cc in range(ccmin, ccmax):
                    indx1 = rr * TS + cc
                    cfa[indx1] = src[row, cc+left] / 65535
                    rgbgreen[indx1] = cfa[indx1]

            # fill lower part
            if rrmax < rr1:
                for rr in range(16):
                    for cc in range(ccmin, ccmax):
                        cfa[(rrmax+rr)*TS+cc] = src[winy+height-rr-2, left+cc] / 65535
                        rgbgreen[(rrmax+rr)*TS+cc] = cfa[(rrmax+rr)*TS+cc]
            
            # fill left border
            if ccmin > 0:
                for rr in range(rrmin, rrmax):
                    for cc in range(16):
                        cfa[rr*TS+cc] = src[row, 32-cc+left] / 65535
                        rgbgreen[rr*TS+cc] = cfa[rr*TS+cc]
            
            # fill right border
            if ccmax < cc1:
                for rr in range(rrmin, rrmax):
                    for cc in range(16):
                        cfa[rr*TS+ccmax+cc] = src[top+rr, winx+width-cc-2] / 65535
                        rgbgreen[rr*TS+ccmax+cc] = cfa[rr*TS+ccmax+cc]

            # fill the image corners
            if rrmin > 0 and ccmin > 0:
                for rr in range(16):
                    for cc in range(16):
                        cfa[rr*TS+cc] = src[winy+32-rr, winx+32-cc] / 65535
                        rgbgreen[rr*TS+cc] = cfa[rr*TS+cc]
            
            if rrmax < rr1 and ccmax < cc1:
                for rr in range(16):
                    for cc in range(16):
                        cfa[(rrmax+rr)*TS+ccmax+cc] = src[winy+height-rr-2, winx+width-cc-2] / 65535
                        rgbgreen[(rrmax+rr)*TS+ccmax+cc] = cfa[(rrmax+rr)*TS+ccmax+cc]
            
            if rrmin > 0 and ccmax < cc1:
                for rr in range(16):
                    for cc in range(16):
                        cfa[rr*TS+ccmax+cc] = src[winy+32-rr, winx+width-cc-2] / 65535
                        rgbgreen[rr*TS+ccmax+cc] = cfa[rr*TS+ccmax+cc]

            if rrmax < rr1 and ccmin > 0:
                for rr in range(16):
                    for cc in range(16):
                        cfa[(rrmax+rr)*TS+cc] = src[winy+height-rr-2, winx+32-cc] / 65535
                        rgbgreen[(rrmax+rr)*TS+cc] = cfa[(rrmax+rr)*TS+cc]
            
            for rr in range(2, rr1-2):
                for cc in range(2, cc1-2):
                    indx = rr*TS+cc
                    delh = abs(cfa[indx + 1] - cfa[indx - 1])
                    delv = abs(cfa[indx + v1] - cfa[indx - v1])
                    dirwTS0[indx] = eps + abs(cfa[indx + v2] - cfa[indx]) + abs(cfa[indx] - cfa[indx - v2]) + delv
                    dirwTS1[indx] = eps + abs(cfa[indx + 2] - cfa[indx]) + abs(cfa[indx] - cfa[indx - 2]) + delh
                    delhvsqsum[indx] = delh ** 2 + delv ** 2

            for rr in range(4, rr1-4):
                fcswitch = fc(cfarray, rr, 4) & 1

                for cc in range(4, cc1-4):
                    indx = rr*TS+cc

                    # colour ratios in each cardinal direction
                    cru = cfa[indx - v1] * (dirwTS0[indx - v2] + dirwTS0[indx]) / (dirwTS0[indx - v2] * (eps + cfa[indx]) + dirwTS0[indx] * (eps + cfa[indx - v2]))
                    crd = cfa[indx + v1] * (dirwTS0[indx + v2] + dirwTS0[indx]) / (dirwTS0[indx + v2] * (eps + cfa[indx]) + dirwTS0[indx] * (eps + cfa[indx + v2]))
                    crl = cfa[indx - 1] * (dirwTS1[indx - 2] + dirwTS1[indx]) / (dirwTS1[indx - 2] * (eps + cfa[indx]) + dirwTS1[indx] * (eps + cfa[indx - 2]))
                    crr = cfa[indx + 1] * (dirwTS1[indx + 2] + dirwTS1[indx]) / (dirwTS1[indx + 2] * (eps + cfa[indx]) + dirwTS1[indx] * (eps + cfa[indx + 2]))

                    # G interpolated in vert/hor directions using Hamilton-Adams method
                    guha = min(clip_pt, cfa[indx - v1] + 0.5 * (cfa[indx] - cfa[indx - v2]))
                    gdha = min(clip_pt, cfa[indx + v1] + 0.5 * (cfa[indx] - cfa[indx + v2]))
                    glha = min(clip_pt, cfa[indx - 1] + 0.5 * (cfa[indx] - cfa[indx - 2]))
                    grha = min(clip_pt, cfa[indx + 1] + 0.5 * (cfa[indx] - cfa[indx + 2]))

                    # G interpolated in vert/hor directions using adaptive ratios
                    guar = cfa[indx] * cru if abs(1-cru) < arthresh else guha
                    gdar = cfa[indx] * crd if abs(1-crd) < arthresh else gdha
                    glar = cfa[indx] * crl if abs(1-crl) < arthresh else glha
                    grar = cfa[indx] * crr if abs(1-crr) < arthresh else grha
                    
                    # adaptive weighTS for vertical/horizontal directions
                    hwt = dirwTS1[indx - 1] / (dirwTS1[indx - 1] + dirwTS1[indx + 1])
                    vwt = dirwTS0[indx - v1] / (dirwTS0[indx + v1] + dirwTS0[indx - v1])

                    # interpolated G via adaptive weighTS of cardinal evaluations
                    Gintvha = vwt * gdha + (1 - vwt) * guha
                    Ginthha = hwt * grha + (1 - hwt) * glha

                    # interpolated colour differences
                    if fcswitch:
                        vcd[indx] = cfa[indx] - (vwt * gdar + (1 - vwt) * guar)
                        hcd[indx] = cfa[indx] - (hwt * grar + (1 - hwt) * glar)
                        vcdalt[indx] = cfa[indx] - Gintvha
                        hcdalt[indx] = cfa[indx] - Ginthha
                    else:
                        vcd[indx] = (vwt * gdar + (1 - vwt) * guar) - cfa[indx]
                        hcd[indx] = (hwt * grar + (1 - hwt) * glar) - cfa[indx]
                        vcdalt[indx] = Gintvha - cfa[indx]
                        hcdalt[indx] = Ginthha - cfa[indx]
                    
                    fcswitch = 1 - fcswitch

                    if cfa[indx] > 0.8 * clip_pt or Gintvha > 0.8 * clip_pt or Ginthha > 0.8 * clip_pt:
                        # use HA if highlighTS are (nearly) clipped
                        guar = guha
                        gdar = gdha
                        glar = glha
                        grar = grha
                        vcd[indx] = vcdalt[indx]
                        hcd[indx] = hcdalt[indx]
                    
                    # differences of interpolations in opposite directions
                    dgintv[indx] = min((guha - gdha) ** 2, (guar - gdar) ** 2)
                    dginth[indx] = min((glha - grha) ** 2, (glar - grar) ** 2)


            for rr in range(4, rr1-4):
                c = fc(cfarray, rr, 4) & 1
                for cc in range(4, cc1-4):
                    indx = rr*TS+cc
                    hcdvar = 3 * (hcd[indx - 2]**2 + hcd[indx]**2 + hcd[indx + 2]**2) - (hcd[indx - 2] + hcd[indx] + hcd[indx + 2])**2
                    hcdaltvar = 3 * (hcdalt[indx - 2]**2 + hcdalt[indx]**2 + hcdalt[indx + 2]**2) - (hcdalt[indx - 2] + hcdalt[indx] + hcdalt[indx + 2])**2
                    vcdvar = 3 * (vcd[indx - v2]**2 + vcd[indx]**2 + vcd[indx + v2]**2) - (vcd[indx - v2] + vcd[indx] + vcd[indx + v2])**2
                    vcdaltvar = 3 * (vcdalt[indx - v2]**2 + vcdalt[indx]**2 + vcdalt[indx + v2]**2) - (vcdalt[indx - v2] + vcdalt[indx] + vcdalt[indx + v2])**2

                    # choose the smallest variance; this yields a smoother interpolation
                    if hcdaltvar < hcdvar:
                        hcd[indx] = hcdalt[indx]
                    if vcdaltvar < vcdvar:
                        vcd[indx] = vcdalt[indx]

                    # bound the interpolation in regions of high saturation
                    # vertical and horizontal G interpolations
                    if c:
                        Ginth = -hcd[indx] + cfa[indx]
                        Gintv = -vcd[indx] + cfa[indx]

                        if hcd[indx] > 0:
                            if 3 * hcd[indx] > (Ginth + cfa[indx]):
                                hcd[indx] = -np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) + cfa[indx]
                            else:
                                hwt = 1 - 3 * hcd[indx] / (eps + Ginth + cfa[indx])
                                hcd[indx] = hwt * hcd[indx] + (1 - hwt) * (-np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) + cfa[indx])

                        if vcd[indx] > 0:
                            if 3 * vcd[indx] > (Gintv + cfa[indx]):
                                vcd[indx] = -np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) + cfa[indx]
                            else:
                                vwt = 1 - 3 * vcd[indx] / (eps + Gintv + cfa[indx])
                                vcd[indx] = vwt * vcd[indx] + (1 - vwt) * (-np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) + cfa[indx])
                        
                        if Ginth > clip_pt:
                            hcd[indx] = -np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) + cfa[indx]

                        if Gintv > clip_pt:
                            vcd[indx] = -np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) + cfa[indx]
                    
                    else:
                        Ginth = hcd[indx] + cfa[indx]
                        Gintv = vcd[indx] + cfa[indx]

                        if hcd[indx] < 0:
                            if 3 * hcd[indx] < -(Ginth + cfa[indx]):
                                hcd[indx] = np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) - cfa[indx]
                            else:
                                hwt = 1 + 3 * hcd[indx] / (eps + Ginth + cfa[indx])
                                hcd[indx] = hwt * hcd[indx] + (1 - hwt) * (np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) - cfa[indx])

                        if vcd[indx] < 0:
                            if 3 * vcd[indx] < -(Gintv + cfa[indx]):
                                vcd[indx] = np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) - cfa[indx]
                            else:
                                vwt = 1 + 3 * vcd[indx] / (eps + Gintv + cfa[indx])
                                vcd[indx] = vwt * vcd[indx] + (1 - vwt) * (np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) - cfa[indx])

                        if Ginth > clip_pt:
                            hcd[indx] = np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) - cfa[indx]

                        if Gintv > clip_pt:
                            vcd[indx] = np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) - cfa[indx]

                        cddiffsq[indx] = (vcd[indx] - hcd[indx])**2

                    c = 1- c

            for rr in range(6, rr1-6):
                for cc in range(6+(fc(cfarray, rr, 2) & 1), cc1-6, 2):
                    indx = rr * TS + cc

                    # compute colour difference variances in cardinal directions
                    uave = vcd[indx] + vcd[indx - v1] + vcd[indx - v2] + vcd[indx - v3]
                    dave = vcd[indx] + vcd[indx + v1] + vcd[indx + v2] + vcd[indx + v3]
                    lave = hcd[indx] + hcd[indx - 1] + hcd[indx - 2] + hcd[indx - 3]
                    rave = hcd[indx] + hcd[indx + 1] + hcd[indx + 2] + hcd[indx + 3]

                    # colour difference (G-R or G-B) variance in up/down/left/right directions
                    Dgrbvvaru = (vcd[indx] - uave)**2 + (vcd[indx - v1] - uave)**2 + (vcd[indx - v2] - uave)**2 + (vcd[indx - v3] - uave)**2
                    Dgrbvvard = (vcd[indx] - dave)**2 + (vcd[indx + v1] - dave)**2 + (vcd[indx + v2] - dave)**2 + (vcd[indx + v3] - dave)**2
                    Dgrbhvarl = (hcd[indx] - lave)**2 + (hcd[indx - 1] - lave)**2 + (hcd[indx - 2] - lave)**2 + (hcd[indx - 3] - lave)**2
                    Dgrbhvarr = (hcd[indx] - rave)**2 + (hcd[indx + 1] - rave)**2 + (hcd[indx + 2] - rave)**2 + (hcd[indx + 3] - rave)**2

                    hwt = dirwTS1[indx - 1] / (dirwTS1[indx - 1] + dirwTS1[indx + 1])
                    vwt = dirwTS0[indx - v1] / (dirwTS0[indx + v1] + dirwTS0[indx - v1])

                    vcdvar = epssq + vwt * Dgrbvvard + (1 - vwt) * Dgrbvvaru
                    hcdvar = epssq + hwt * Dgrbhvarr + (1 - hwt) * Dgrbhvarl

                    # compute fluctuations in up/down and left/right interpolations of colours
                    Dgrbvvaru = (dgintv[indx]) + (dgintv[indx - v1]) + (dgintv[indx - v2])
                    Dgrbvvard = (dgintv[indx]) + (dgintv[indx + v1]) + (dgintv[indx + v2])
                    Dgrbhvarl = (dginth[indx]) + (dginth[indx - 1]) + (dginth[indx - 2])
                    Dgrbhvarr = (dginth[indx]) + (dginth[indx + 1]) + (dginth[indx + 2])

                    vcdvar1 = epssq + vwt * Dgrbvvard + (1 - vwt) * Dgrbvvaru
                    hcdvar1 = epssq + hwt * Dgrbhvarr + (1 - hwt) * Dgrbhvarl

                    # determine adaptive weighTS for G interpolation
                    varwt = hcdvar / (vcdvar + hcdvar)
                    diffwt = hcdvar1 / (vcdvar1 + hcdvar1)

                    # if both agree on interpolation direction, choose the one with strongest directional discrimination;
                    # otherwise, choose the u/d and l/r difference fluctuation weighTS
                    if ((0.5 - varwt) * (0.5 - diffwt) > 0) and (abs(0.5 - diffwt) < abs(0.5 - varwt)):
                        hvwt[indx >> 1] = varwt
                    else:
                        hvwt[indx >> 1] = diffwt

            for rr in range(6, rr1-6):
                for cc in range(6 + (fc(cfarray, rr, 2) & 1), cc1 - 6, 2):
                    indx = rr * TS + cc
                    nyqutest[indx >> 1] = (gaussodd[0] * cddiffsq[indx] + gaussodd[1] * (cddiffsq[(indx - m1)] + cddiffsq[(indx + p1)] + cddiffsq[(indx - p1)] + cddiffsq[(indx + m1)]) + gaussodd[2] * (cddiffsq[(indx - v2)] + cddiffsq[(indx - 2)] + cddiffsq[(indx + 2)] + cddiffsq[(indx + v2)]) + gaussodd[3] * (cddiffsq[(indx - m2)] + cddiffsq[(indx + p2)] + cddiffsq[(indx - p2)] + cddiffsq[(indx + m2)])) - (gaussgrad[0] *  delhvsqsum[indx] + gaussgrad[1] * (delhvsqsum[indx - v1] + delhvsqsum[indx + 1] + delhvsqsum[indx - 1] + delhvsqsum[indx + v1]) + gaussgrad[2] * (delhvsqsum[indx - m1] + delhvsqsum[indx + p1] + delhvsqsum[indx - p1] + delhvsqsum[indx + m1]) + gaussgrad[3] * (delhvsqsum[indx - v2] + delhvsqsum[indx - 2] + delhvsqsum[indx + 2] + delhvsqsum[indx + v2]) + gaussgrad[4] * (delhvsqsum[indx - v2 - 1] + delhvsqsum[indx - v2 + 1] + delhvsqsum[indx - TS - 2] + delhvsqsum[indx - TS + 2] + delhvsqsum[indx + TS - 2] + delhvsqsum[indx + TS + 2] + delhvsqsum[indx + v2 - 1] + delhvsqsum[indx + v2 + 1]) + gaussgrad[5] * (delhvsqsum[indx - m2] + delhvsqsum[indx + p2] + delhvsqsum[indx - p2] + delhvsqsum[indx + m2]))
            
            # Nyquist test
            nystartrow = 0
            nyendrow = 0
            nystartcol = TS + 1
            nyendcol = 0

            for rr in range(6, rr1-6):
                for cc in range(6 + (fc(cfarray, rr, 2) & 1), cc1 - 6, 2):
                    indx = rr * TS + cc
                    # nyquist texture test: ask if difference of vcd compared to hcd is larger or smaller than RGGB gradienTS
                    if nyqutest[indx >> 1] > 0:
                        nyquist[indx >> 1] = 1;    # nyquist=1 for nyquist region
                        nystartrow = nystartrow if nystartrow else rr
                        nyendrow = rr
                        nystartcol = cc if nystartcol > cc else nystartcol
                        nyendcol = cc if nyendcol < cc else nyendcol

            doNyquist = (nystartrow != nyendrow) and (nystartcol != nyendcol)

            if doNyquist:
                nyendrow += 1
                nyendcol += 1
                nystartcol -= (nystartcol & 1)
                nystartrow = max(8, nystartrow)
                nyendrow = min(rr1 - 8, nyendrow)
                nystartcol = max(8, nystartcol)
                nyendcol = min(cc1 - 8, nyendcol)
                nyquist2 = np.zeros(TS*TS, dtype=np.int16)

                for rr in range(nystartrow, nyendrow):
                    for indx in range(rr*TS+nystartcol+(fc(cfarray, rr, 2)&1), rr*TS+nyendcol, 2):
                        nyquisttemp = (nyquist[(indx - v2) >> 1] + nyquist[(indx - m1) >> 1] + nyquist[(indx + p1) >> 1] + nyquist[(indx - 2) >> 1] + nyquist[(indx + 2) >> 1] + nyquist[(indx - p1) >> 1] + nyquist[(indx + m1) >> 1] + nyquist[(indx + v2) >> 1])
                        # if most of your neighbours are named Nyquist, it's likely that you're one too, or not
                        nyquist2[indx >> 1] = 1 if nyquisttemp > 4 else (0 if nyquisttemp < 4 else nyquist[indx >> 1])
                
                # end of Nyquist test

                # in areas of Nyquist texture, do area interpolation

                for rr in range(nystartrow, nyendrow):
                    for indx in range(rr * TS + nystartcol + (fc(cfarray, rr, 2)&1), rr * TS + nyendcol, 2):
                        if nyquist2[indx >> 1]:
                            # area interpolation
                            sumcfa = sumh = sumv = sumsqh = sumsqv = areawt = 0

                            for i in range(-6, 7, 2):
                                for j in range(-6, 7, 2):
                                    indx1 = indx + (i * TS) + j
                                    if nyquist2[indx1 >> 1]:
                                        cfatemp = cfa[indx1]
                                        sumcfa += cfatemp
                                        sumh += (cfa[indx1 - 1] + cfa[indx1 + 1])
                                        sumv += (cfa[indx1 - v1] + cfa[indx1 + v1])
                                        sumsqh += (cfatemp - cfa[indx1 - 1])**2 + (cfatemp - cfa[indx1 + 1])**2
                                        sumsqv += (cfatemp - cfa[indx1 - v1])**2 + (cfatemp - cfa[indx1 + v1])**2
                                        areawt += 1
                            
                            # horizontal and vertical colour differences, and adaptive weight
                            hcdvar = epssq + max(0, areawt*sumsqh - sumh*sumh)
                            vcdvar = epssq + max(0, areawt*sumsqv - sumv*sumv)
                            hvwt[indx] = hcdvar / (vcdvar+hcdvar)

                            # end of area interpolation
            
            # populate G at R/B sites
            for rr in range(8, rr1-8):
                for indx in range(rr * TS + 8 + (fc(cfarray, rr, 2) & 1), rr * TS + cc1 - 8, 2):
                    # first ask if one geTS more directional discrimination from nearby B/R sites
                    hvwtalt = 0.25*(hvwt[(indx-m1)>>1]+hvwt[(indx+p1)>>1]+hvwt[(indx-p1)>>1]+hvwt[(indx+m1)>>1])
                    vo=abs(0.5-hvwt[indx>>1])
                    ve=abs(0.5-hvwtalt)
                    if vo < ve:
                        hvwt[indx>>1] = hvwtalt # a better result was obtained from the neighbors

                    # evaluate colour differences   
                    Dgrb[0, indx>>1] = intp(hvwt[indx >> 1], vcd[indx], hcd[indx])
                    # evaluate G
                    rgbgreen[indx] = cfa[indx] + Dgrb[0, indx >> 1]
                    # local curvature in G (preparation for nyquist refinement step)
                    if nyquist2[indx >> 1]:
                        Dgrbh2[indx >> 1] = (rgbgreen[indx] - 0.5*(rgbgreen[indx-1]+rgbgreen[indx+1]))**2
                        Dgrbh2[indx >> 1] = (rgbgreen[indx] - 0.5*(rgbgreen[indx-v1]+rgbgreen[indx+v1]))**2
                    else:
                        Dgrbh2[indx >> 1] = Dgrbh2[indx >> 1] = 0

            # end of standard interpolation

            # refine Nyquist areas using G curvatures
            if doNyquist:
                for rr in range(nystartrow, nyendrow):
                    for indx in range(rr * TS + nystartcol + (fc(cfarray, rr, 2) & 1), rr * TS + nyendcol, 2):
                        if nyquist2[indx >> 1]:
                            # local averages (over Nyquist pixels only) of G curvature squared
                            gvarh = epssq + (gquinc[0] * Dgrbh2[indx >> 1] + gquinc[1] * (Dgrbh2[(indx - m1) >> 1] + Dgrbh2[(indx + p1) >> 1] + Dgrbh2[(indx - p1) >> 1] + Dgrbh2[(indx + m1) >> 1]) + gquinc[2] * (Dgrbh2[(indx - v2) >> 1] + Dgrbh2[(indx - 2) >> 1] + Dgrbh2[(indx + 2) >> 1] + Dgrbh2[(indx + v2) >> 1]) + gquinc[3] * (Dgrbh2[(indx - m2) >> 1] + Dgrbh2[(indx + p2) >> 1] + Dgrbh2[(indx - p2) >> 1] + Dgrbh2[(indx + m2) >> 1]))
                            gvarv = epssq + (gquinc[0] * Dgrbv2[indx >> 1] + gquinc[1] * (Dgrbv2[(indx - m1) >> 1] + Dgrbv2[(indx + p1) >> 1] + Dgrbv2[(indx - p1) >> 1] + Dgrbv2[(indx + m1) >> 1]) + gquinc[2] * (Dgrbv2[(indx - v2) >> 1] + Dgrbv2[(indx - 2) >> 1] + Dgrbv2[(indx + 2) >> 1] + Dgrbv2[(indx + v2) >> 1]) + gquinc[3] * (Dgrbv2[(indx - m2) >> 1] + Dgrbv2[(indx + p2) >> 1] + Dgrbv2[(indx - p2) >> 1] + Dgrbv2[(indx + m2) >> 1]))
                            # use the resulTS as weighTS for refined G interpolation
                            Dgrb[0, indx >> 1] = (hcd[indx] * gvarv + vcd[indx] * gvarh) / (gvarv + gvarh)
                            rgbgreen[indx] = cfa[indx] + Dgrb[0, indx >> 1]
            
            for rr in range(6, rr1-6):
                if (fc(cfarray, rr, 2) & 1) == 0:
                    for cc in range(6, cc1-6, 2):
                        indx = rr * TS + cc
                        delp[indx >> 1] = abs(cfa[indx + p1] - cfa[indx - p1])
                        delm[indx >> 1] = abs(cfa[indx + m1] - cfa[indx - m1])
                        Dgrbsq1p[indx >> 1] = ((cfa[indx + 1] - cfa[indx + 1 - p1])**2 + (cfa[indx + 1] - cfa[indx + 1 + p1])**2)
                        Dgrbsq1m[indx >> 1] = ((cfa[indx + 1] - cfa[indx + 1 - m1])**2 + (cfa[indx + 1] - cfa[indx + 1 + m1])**2)
                else:
                    for cc in range(6, cc1-6, 2):
                        indx = rr * TS + cc
                        delp[indx >> 1] = abs(cfa[indx + 1 + p1] - cfa[indx + 1 - p1])
                        delm[indx >> 1] = abs(cfa[indx + 1 + m1] - cfa[indx + 1 - m1])
                        Dgrbsq1p[indx >> 1] = ((cfa[indx] - cfa[indx - p1])**2 + (cfa[indx] - cfa[indx + p1])**2)
                        Dgrbsq1m[indx >> 1] = ((cfa[indx] - cfa[indx - m1])**2 + (cfa[indx] - cfa[indx + m1])**2)

            # diagonal interpolation correction
            for rr in range(8, rr1-8):
                for cc in range(8 + (fc(cfarray, rr, 2) & 1), cc1 - 8, 2):
                    indx = rr * TS + cc
                    indx1 = indx >> 1

                    # diagonal colour ratios
                    crse = 2 * (cfa[indx + m1]) / (eps + cfa[indx] + (cfa[indx + m2]))
                    crnw = 2 * (cfa[indx - m1]) / (eps + cfa[indx] + (cfa[indx - m2]))
                    crne = 2 * (cfa[indx + p1]) / (eps + cfa[indx] + (cfa[indx + p2]))
                    crsw = 2 * (cfa[indx - p1]) / (eps + cfa[indx] + (cfa[indx - p2]))

                    # colour differences in diagonal directions
                    # assign B/R at R/B sites
                    if abs(1 - crse) < arthresh:
                        rbse = cfa[indx] * crse
                    else:
                        rbse = cfa[indx + m1] + 0.5 * (cfa[indx] - cfa[indx + m2])

                    if abs(1 - crnw) < arthresh:
                        rbnw = (cfa[indx - m1]) + 0.5 *(cfa[indx] - cfa[indx - m2])

                    if abs(1 - crne) < arthresh:
                        rbne = cfa[indx] * crne
                    else:
                        rbne = (cfa[indx + p1]) + 0.5 * cfa[indx] - cfa[indx + p2]

                    if abs(1 - crsw) < arthresh:
                        rbsw = cfa[indx] * crsw
                    else:
                        rbsw = (cfa[indx - p1]) + 0.5 * (cfa[indx] - cfa[indx - p2])
                    
                    wTSe = eps + delm[indx1] + delm[(indx + m1) >> 1] + delm[(indx + m2) >> 1] # same as for wtu,wtd,wtl,wtr
                    wtnw = eps + delm[indx1] + delm[(indx - m1) >> 1] + delm[(indx - m2) >> 1]
                    wtne = eps + delp[indx1] + delp[(indx + p1) >> 1] + delp[(indx + p2) >> 1]
                    wTSw = eps + delp[indx1] + delp[(indx - p1) >> 1] + delp[(indx - p2) >> 1]

                    rbm[indx1] = (wTSe * rbnw + wtnw * rbse) / (wTSe + wtnw)
                    rbp[indx1] = (wtne * rbsw + wTSw * rbne) / (wtne + wTSw)

                    # variance of R-B in plus/minus directions
                    rbvarm = epssq + (gausseven[0] * (Dgrbsq1m[(indx - v1) >> 1] + Dgrbsq1m[(indx - 1) >> 1] + Dgrbsq1m[(indx + 1) >> 1] + Dgrbsq1m[(indx + v1) >> 1]) + gausseven[1] * (Dgrbsq1m[(indx - v2 - 1) >> 1] + Dgrbsq1m[(indx - v2 + 1) >> 1] + Dgrbsq1m[(indx - 2 - v1) >> 1] + Dgrbsq1m[(indx + 2 - v1) >> 1] + Dgrbsq1m[(indx - 2 + v1) >> 1] + Dgrbsq1m[(indx + 2 + v1) >> 1] + Dgrbsq1m[(indx + v2 - 1) >> 1] + Dgrbsq1m[(indx + v2 + 1) >> 1]))

                    pmwt[indx1] = rbvarm / ((epssq + (gausseven[0] * (Dgrbsq1p[(indx - v1) >> 1] + Dgrbsq1p[(indx - 1) >> 1] + Dgrbsq1p[(indx + 1) >> 1] + Dgrbsq1p[(indx + v1) >> 1]) + gausseven[1] * (Dgrbsq1p[(indx - v2 - 1) >> 1] + Dgrbsq1p[(indx - v2 + 1) >> 1] + Dgrbsq1p[(indx - 2 - v1) >> 1] + Dgrbsq1p[(indx + 2 - v1) >> 1] + Dgrbsq1p[(indx - 2 + v1) >> 1] + Dgrbsq1p[(indx + 2 + v1) >> 1] + Dgrbsq1p[(indx + v2 - 1) >> 1] + Dgrbsq1p[(indx + v2 + 1) >> 1]))) + rbvarm)

                    # bound the interpolation in regions of high saturation
                    if rbp[indx1] < cfa[indx]:
                        if 2 * (rbp[indx1]) < cfa[indx]:
                            rbp[indx1] = np.median([rbp[indx1] , cfa[indx - p1], cfa[indx + p1]])
                        else:
                            pwt = 2 * (cfa[indx] - rbp[indx1]) / (eps + rbp[indx1] + cfa[indx])
                            rbp[indx1] = pwt * rbp[indx1] + (1 - pwt) * np.median([rbp[indx1], cfa[indx - p1], cfa[indx + p1]])

                    if rbm[indx1] < cfa[indx]:
                        if 2 * (rbm[indx1]) < cfa[indx]:
                            rbm[indx1] = np.median([rbm[indx1] , cfa[indx - m1], cfa[indx + m1]])
                        else:
                            mwt = 2 * (cfa[indx] - rbm[indx1]) / (eps + rbm[indx1] + cfa[indx])
                            rbm[indx1] = mwt * rbm[indx1] + (1 - mwt) * np.median([rbm[indx1], cfa[indx - m1], cfa[indx + m1]])

                    if rbp[indx1] > clip_pt:
                        rbp[indx1] = np.median([rbp[indx1], cfa[indx - p1], cfa[indx + p1]])

                    if rbm[indx1] > clip_pt:
                        rbm[indx1] = np.median([rbm[indx1], cfa[indx - m1], cfa[indx + m1]])
            

            for rr in range(10, rr1-10):
                for cc in range(10 + (fc(cfarray, rr, 2) & 1), cc1-10, 2):
                    indx = rr * TS + cc
                    indx1 = indx >> 1
                    # first ask if one geTS more directional discrimination from nearby B/R sites
                    pmwtalt = 0.25 * (pmwt[(indx - m1) >> 1] + pmwt[(indx + p1) >> 1] + pmwt[(indx - p1) >> 1] + pmwt[(indx + m1) >> 1])
                    if abs(0.5 - pmwt[indx1]) < abs(0.5 - pmwtalt):
                        pmwt[indx1] = pmwtalt   # a better result was obtained from the neighbours

                    rbint[indx1] = 0.5 * (cfa[indx] + rbm[indx1] * (1 - pmwt[indx1]) + rbp[indx1] * pmwt[indx1])  # this is R+B, interpolated
            
            for rr in range(12, rr1 - 12):
                for cc in range(12 + (fc(cfarray, rr, 2) & 1), cc1 - 12, 2):

                    if abs(0.5 - pmwt[indx >> 1]) < abs(0.5 - hvwt[indx >> 1]):
                        continue

                    # now interpolate G vertically/horizontally using R+B values
                    # unfortunately, since G interpolation cannot be done diagonally this may lead to colour shifTS

                    # colour ratios for G interpolation
                    cru = cfa[indx - v1] * 2 / (eps + rbint[indx1] + rbint[(indx1 - v1)])
                    crd = cfa[indx + v1] * 2 / (eps + rbint[indx1] + rbint[(indx1 + v1)])
                    crl = cfa[indx - 1] * 2 / (eps + rbint[indx1] + rbint[(indx1 - 1)])
                    crr = cfa[indx + 1] * 2 / (eps + rbint[indx1] + rbint[(indx1 + 1)])

                    # interpolated G via adaptive ratios or Hamilton-Adams in each cardinal direction
                    if abs(1 - cru) < arthresh:
                        gu = rbint[indx1] * cru
                    else:
                        gu = cfa[indx - v1] + 0.5 * (rbint[indx1] - rbint[(indx1 - v1)])

                    if abs(1 - crd) < arthresh:
                        gd = rbint[indx1] * crd
                    else:
                        gd = cfa[indx + v1] + 0.5 * (rbint[indx1] - rbint[(indx1 + v1)])

                    if abs(1 - crl) < arthresh:
                        gl = rbint[indx1] * crl
                    else:
                        gl = cfa[indx - 1] + 0.5 * (rbint[indx1] - rbint[(indx1 - 1)])

                    if abs(1 - crr) < arthresh:
                        gr = rbint[indx1] * crr
                    else:
                        gr = cfa[indx + 1] + 0.5 * (rbint[indx1] - rbint[(indx1 + 1)])

                    # interpolated G via adaptive weighTS of cardinal evaluations
                    Gintv = (dirwTS0[indx - v1] * gd + dirwTS0[indx + v1] * gu) / (dirwTS0[indx + v1] + dirwTS0[indx - v1])
                    Ginth = (dirwTS1[indx - 1] * gr + dirwTS1[indx + 1] * gl) / (dirwTS1[indx - 1] + dirwTS1[indx + 1])

                    # bound the interpolation in regions of high saturation
                    if Gintv < rbint[indx1]:
                        if (2 * Gintv < rbint[indx1]):
                            Gintv = np.median([Gintv , cfa[indx - v1], cfa[indx + v1]])
                        else:
                            vwt = 2 * (rbint[indx1] - Gintv) / (eps + Gintv + rbint[indx1])
                            Gintv = vwt * Gintv + (1 - vwt) * np.median([Gintv, cfa[indx - v1], cfa[indx + v1]])

                    if Ginth < rbint[indx1]:
                        if 2 * Ginth < rbint[indx1]:
                            Ginth = np.median([Ginth , cfa[indx - 1], cfa[indx + 1]])
                        else:
                            hwt = 2 * (rbint[indx1] - Ginth) / (eps + Ginth + rbint[indx1])
                            Ginth = hwt * Ginth + (1 - hwt) * np.median([Ginth, cfa[indx - 1], cfa[indx + 1]])

                    if Ginth > clip_pt:
                        Ginth = np.median([Ginth, cfa[indx - 1], cfa[indx + 1]])

                    if Gintv > clip_pt:
                        Gintv = np.median([Gintv, cfa[indx - v1], cfa[indx + v1]])

                    rgbgreen[indx] = Ginth * (1 - hvwt[indx1]) + Gintv * hvwt[indx1]
                    Dgrb[0, indx >> 1] = rgbgreen[indx] - cfa[indx]
            
            # end of diagonal interpolation correction

            # fancy chrominance interpolation
            # (ey,ex) is location of R site
            for rr in range(13-ey, rr1-12, 2):
                for indx1 in range((rr * TS + 13 - ex) >> 1, (rr * TS + cc1 - 12) >> 1):
                    # B coset
                    Dgrb[1, indx1] = Dgrb[0, indx1]; # split out G-B from G-R
                    Dgrb[0, indx1] = 0

            for rr in range(14, rr1 - 14):
                c = int(1 - fc(cfarray, rr, 14 + (fc(cfarray, rr, 2) & 1)) / 2)
                for cc in range(14 + (fc(cfarray, rr, 2) & 1), cc1 - 14, 2):
                    indx = rr * TS + cc
                    wtnw = 1 / (eps + abs(Dgrb[c, (indx - m1) >> 1] - Dgrb[c, (indx + m1) >> 1]) + abs(Dgrb[c, (indx - m1) >> 1] - Dgrb[c, (indx - m3) >> 1]) + abs(Dgrb[c, (indx + m1) >> 1] - Dgrb[c, (indx - m3) >> 1]))
                    wtne = 1 / (eps + abs(Dgrb[c, (indx + p1) >> 1] - Dgrb[c, (indx - p1) >> 1]) + abs(Dgrb[c, (indx + p1) >> 1] - Dgrb[c, (indx + p3) >> 1]) + abs(Dgrb[c, (indx - p1) >> 1] - Dgrb[c, (indx + p3) >> 1]))
                    wTSw = 1 / (eps + abs(Dgrb[c, (indx - p1) >> 1] - Dgrb[c, (indx + p1) >> 1]) + abs(Dgrb[c, (indx - p1) >> 1] - Dgrb[c, (indx + m3) >> 1]) + abs(Dgrb[c, (indx + p1) >> 1] - Dgrb[c, (indx - p3) >> 1]))
                    wTSe = 1 / (eps + abs(Dgrb[c, (indx + m1) >> 1] - Dgrb[c, (indx - m1) >> 1]) + abs(Dgrb[c, (indx + m1) >> 1] - Dgrb[c, (indx - p3) >> 1]) + abs(Dgrb[c, (indx - m1) >> 1] - Dgrb[c, (indx + m3) >> 1]))

                    Dgrb[c, indx >> 1] = (wtnw * (1.325 * Dgrb[c, (indx - m1) >> 1] - 0.175 * Dgrb[c, (indx - m3) >> 1] - 0.075 * Dgrb[c, (indx - m1 - 2) >> 1] - 0.075 * Dgrb[c, (indx - m1 - v2) >> 1] ) + wtne * (1.325 * Dgrb[c, (indx + p1) >> 1] - 0.175 * Dgrb[c, (indx + p3) >> 1] - 0.075 * Dgrb[c, (indx + p1 + 2) >> 1] - 0.075 * Dgrb[c, (indx + p1 + v2) >> 1] ) + wTSw * (1.325 * Dgrb[c, (indx - p1) >> 1] - 0.175 * Dgrb[c, (indx - p3) >> 1] - 0.075 * Dgrb[c, (indx - p1 - 2) >> 1] - 0.075 * Dgrb[c, (indx - p1 - v2) >> 1] ) + wTSe * (1.325 * Dgrb[c, (indx + m1) >> 1] - 0.175 * Dgrb[c, (indx + m3) >> 1] - 0.075 * Dgrb[c, (indx + m1 + 2) >> 1] - 0.075 * Dgrb[c, (indx + m1 + v2) >> 1] )) / (wtnw + wtne + wTSw + wTSe)

            for rr in range(16, rr1-16):
                row = rr + top
                col = left + 16

                if (fc(cfarray, rr, 2) & 1) == 1:

                    for indx in range(rr * TS + 16, rr * TS + cc1 - 16 - (cc1 & 1), 2):

                        temp =  1 / (hvwt[(indx - v1) >> 1] + 2 - hvwt[(indx + 1) >> 1] - hvwt[(indx - 1) >> 1] + hvwt[(indx + v1) >> 1])

                        RGB[row, col, 0] = max(0, 65535 * (rgbgreen[indx] - ((hvwt[(indx - v1) >> 1]) * Dgrb[0, (indx - v1) >> 1] + (1 - hvwt[(indx + 1) >> 1]) * Dgrb[0, (indx + 1) >> 1] + (1 - hvwt[(indx - 1) >> 1]) * Dgrb[0, (indx - 1) >> 1] + (hvwt[(indx + v1) >> 1]) * Dgrb[0, (indx + v1) >> 1]) * temp))

                        RGB[row, col, 2] = max(0, 65535 * (rgbgreen[indx] - ((hvwt[(indx - v1) >> 1]) * Dgrb[1, (indx - v1) >> 1] + (1 - hvwt[(indx + 1) >> 1]) * Dgrb[1, (indx + 1) >> 1] + (1 - hvwt[(indx - 1) >> 1]) * Dgrb[1, (indx - 1) >> 1] + (hvwt[(indx + v1) >> 1]) * Dgrb[1, (indx + v1) >> 1]) * temp))

                        indx += 1
                        col += 1
                        
                        RGB[row, col, 0] = max(0, 65535 * (rgbgreen[indx] - Dgrb[0, indx >> 1]))
                        RGB[row, col, 2] = max(0, 65535 * (rgbgreen[indx] - Dgrb[1, indx >> 1]))

                        col += 1

                    if cc1 & 1: # width of tile is odd

                        temp =  1 / (hvwt[(indx - v1) >> 1] + 2 - hvwt[(indx + 1) >> 1] - hvwt[(indx - 1) >> 1] + hvwt[(indx + v1) >> 1])

                        RGB[row, col, 0] = max(0, 65535 * (rgbgreen[indx] - ((hvwt[(indx - v1) >> 1]) * Dgrb[0, (indx - v1) >> 1] + (1 - hvwt[(indx + 1) >> 1]) * Dgrb[0, (indx + 1) >> 1] + (1 - hvwt[(indx - 1) >> 1]) * Dgrb[0, (indx - 1) >> 1] + (hvwt[(indx + v1) >> 1]) * Dgrb[0, (indx + v1) >> 1]) * temp))

                        RGB[row, col, 2] = max(0, 65535 * (rgbgreen[indx] - ((hvwt[(indx - v1) >> 1]) * Dgrb[1, (indx - v1) >> 1] + (1 - hvwt[(indx + 1) >> 1]) * Dgrb[1, (indx + 1) >> 1] + (1 - hvwt[(indx - 1) >> 1]) * Dgrb[1, (indx - 1) >> 1] + (hvwt[(indx + v1) >> 1]) * Dgrb[1, (indx + v1) >> 1]) * temp))
                
                else:

                    for indx in range(rr * TS + 16, rr * TS + cc1 - 16 - (cc1 & 1), 2):

                        RGB[row, col, 0] = max(0, 65535 * (rgbgreen[indx] - Dgrb[0, indx >> 1]))
                        RGB[row, col, 2] = max(0, 65535 * (rgbgreen[indx] - Dgrb[1, indx >> 1]))

                        indx += 1
                        col += 1

                        temp =  1 / (hvwt[(indx - v1) >> 1] + 2 - hvwt[(indx + 1) >> 1] - hvwt[(indx - 1) >> 1] + hvwt[(indx + v1) >> 1])
                        RGB[row, col, 0] = max(0, 65535 * (rgbgreen[indx] - ((hvwt[(indx - v1) >> 1]) * Dgrb[0, (indx - v1) >> 1] + (1 - hvwt[(indx + 1) >> 1]) * Dgrb[0, (indx + 1) >> 1] + (1 - hvwt[(indx - 1) >> 1]) * Dgrb[0, (indx - 1) >> 1] + (hvwt[(indx + v1) >> 1]) * Dgrb[0, (indx + v1) >> 1]) * temp))
                        RGB[row, col, 2] = max(0, 65535 * (rgbgreen[indx] - ((hvwt[(indx - v1) >> 1]) * Dgrb[1, (indx - v1) >> 1] + (1 - hvwt[(indx + 1) >> 1]) * Dgrb[1, (indx + 1) >> 1] + (1 - hvwt[(indx - 1) >> 1]) * Dgrb[1, (indx - 1) >> 1] + (hvwt[(indx + v1) >> 1]) * Dgrb[1, (indx + v1) >> 1]) * temp))

                        col += 1

                    if cc1 & 1: # width of tile is odd
                        RGB[row, col, 0] = max(0, 65535 * (rgbgreen[indx] - Dgrb[0, indx >> 1]))
                        RGB[row, col, 2] = max(0, 65535 * (rgbgreen[indx] - Dgrb[1, indx >> 1]))

            # copy smoothed resulTS back to image matrix
            for rr in range(16, rr1 - 16):
                row = rr + top

                for cc in range(16, cc1-16):
                    RGB[row, cc+left, 1] = max(0, 65535 * rgbgreen[rr * TS + cc])

    return RGB

def amaze_demosaic_libraw(src, cfarray, daylight_wb):

    TS = 512
    winx = winy = 0
    width = src.shape[1]
    height = src.shape[0]
    image = np.empty([height, width, 3], dtype=np.uint16)
    clip_pt = min(daylight_wb[0], daylight_wb[1], daylight_wb[2])

    v1 = TS
    v2 = 2 * TS
    v3 = 3 * TS
    p1 = -TS + 1
    p2 = -2 * TS + 2
    p3 = -3 * TS + 3
    m1 = TS + 1 
    m2 = 2 * TS + 2
    m3 = 3 * TS + 3

    nbr = [-v2,-2,2,v2,0]
    eps, epssq = 1e-5, 1e-10

    # adaptive ratios threshold
    arthresh=0.75
    # nyquist texture test threshold
    nyqthresh=0.5
    # diagonal interpolation test threshold
    pmthresh=0.25
    # factors for bounding interpolation in saturated regions
    lbd, ubd = 1, 1 # lbd=0.66, ubd=1.5 alternative values;

    # gaussian on 5x5 quincunx, sigma=1.2
    gaussodd = [0.14659727707323927, 0.103592713382435, 0.0732036125103057, 0.0365543548389495]
    # gaussian on 5x5, sigma=1.2
    gaussgrad = [0.07384411893421103, 0.06207511968171489, 0.0521818194747806, 0.03687419286733595, 0.03099732204057846, 0.018413194161458882]
    # gaussian on 3x3, sigma =0.7
    gauss1 = [0.3376688223162362, 0.12171198028231786, 0.04387081413862306]
    # gaussian on 5x5 alt quincunx, sigma=1.5
    gausseven = [0.13719494435797422, 0.05640252782101291]
    # guassian on quincunx grid
    gquinc = [0.169917, 0.108947, 0.069855, 0.0287182]

    progress = 0.0

    rgb = np.empty([TS*TS, 3], dtype=np.float32)
    delh = np.empty(TS*TS, dtype=np.float32)
    delv = np.empty(TS*TS, dtype=np.float32)
    delhsq = np.empty(TS*TS, dtype=np.float32)
    delvsq = np.empty(TS*TS, dtype=np.float32)
    dirwts = np.empty([TS*TS, 2], dtype=np.float32)
    vcd = np.empty(TS*TS, dtype=np.float32)
    hcd = np.empty(TS*TS, dtype=np.float32)
    vcdalt = np.empty(TS*TS, dtype=np.float32)
    hcdalt = np.empty(TS*TS, dtype=np.float32)
    vcdsq = np.empty(TS*TS, dtype=np.float32)
    hcdsq = np.empty(TS*TS, dtype=np.float32)
    cddiffsq = np.empty(TS*TS, dtype=np.float32)
    hvwt = np.empty(TS*TS, dtype=np.float32)
    Dgrb = np.empty([TS*TS, 2], dtype=np.float32)
    delp = np.empty(TS*TS, dtype=np.float32)
    delm = np.empty(TS*TS, dtype=np.float32)
    rbint = np.empty(TS*TS, dtype=np.float32)
    Dgrbh2 = np.empty(TS*TS, dtype=np.float32)
    Dgrbv2 = np.empty(TS*TS, dtype=np.float32)
    dgintv = np.empty(TS*TS, dtype=np.float32)
    dginth = np.empty(TS*TS, dtype=np.float32)
    Dgrbp1 = np.empty(TS*TS, dtype=np.float32)
    Dgrbm1 = np.empty(TS*TS, dtype=np.float32)
    Dgrbpsq1 = np.empty(TS*TS, dtype=np.float32)
    Dgrbmsq1 = np.empty(TS*TS, dtype=np.float32)
    cfa = np.empty(TS*TS, dtype=np.float32)
    pmwt = np.empty(TS*TS, dtype=np.float32)
    rbp = np.empty(TS*TS, dtype=np.float32)
    rbm = np.empty(TS*TS, dtype=np.float32)

    nyquist = np.empty(TS*TS, dtype=np.int32)

    # determine GRBG coset; (ey,ex) is the offset of the R subarray
    if fc(cfarray, 0, 0)== 1:
        if fc(cfarray, 0, 1) == 0:
            ex, ey = 1, 0
        else:
            ex, ey = 0, 1
    else:
        if fc(cfarray, 0, 0) == 0:
            ex = ey = 0
        else: 
            ex = ey = 1
    
    # Start main loop
    loop_cnt = 1    
    for top in range(winy-16, winy+height, TS-32):
        for left in range(winx-16, winx+width, TS-32):
            print("Loop [{}]: top: {} left: {}".format(loop_cnt, top, left))
            loop_cnt += 1
            # location of tile bottom edge
            bottom = min(top+TS, winy+height+16)
            # location of tile right edge
            right = min(left+TS, winx+width+16)
            # tile width  (=TS except for right edge of image)
            rr1 = bottom - top
            # tile height (=TS except for bottom edge of image)
            cc1 = right - left
            
            # rgb from input CFA data
            # rgb values should be floating point number between 0 and 1 
            # after white balance multipliers are applied 
            # a 16 pixel border is added to each side of the image

            # bookkeeping for borders
            rrmin = 16 if top < winy else 0
            ccmin = 16 if left < winx else 0
            rrmax = winy+height-top if bottom>(winy+height) else rr1
            ccmax = winx+width-left if right>(winx+width) else cc1

            for rr in range(rrmin, rrmax):
                row = rr + top
                for cc in range(ccmin, ccmax):
                    col = cc + left
                    c = cfarray[rr, cc]
                    indx1 = rr * TS + cc
                    indx = row * width + col
                    rgb[indx1, c] = src[row, col] / 65535

                    cfa[indx1] = rgb[indx1, c]
            
            # fill borders
            if rrmin > 0:
                for rr in range(16):
                    for cc in range(ccmin, ccmax):
                        c = cfarray[rr, cc]
                        rgb[rr*TS+cc, c] = rgb[(32-rr)*TS+cc, c]
                        cfa[rr*TS+cc] = rgb[rr*TS+cc, c]
            
            if rrmax < rr1:
                for rr in range(16):
                    for cc in range(ccmin, ccmax):
                        c = cfarray[rr, cc]
                        rgb[(rrmax+rr)*TS+cc, c] = (src[(winy+height-rr-2), left+cc])/65535
                        cfa[(rrmax+rr)*TS+cc] = rgb[(rrmax+rr)*TS+cc, c]
            
            if ccmin > 0:
                for rr in range(rrmin, rrmax):
                    for cc in range(16):
                        c = cfarray[rr, cc]
                        rgb[rr*TS+cc, c] = rgb[rr*TS+32-cc, c]
                        cfa[rr*TS+cc] = rgb[rr*TS+cc, c]
            
            if ccmax < cc1:
                for rr in range(rrmin, rrmax):
                    for cc in range(16):
                        c = cfarray[rr, cc]
                        rgb[rr*TS+ccmax+cc, c] = (src[(top+rr), (winx+width-cc-2)])/65535
                        cfa[rr*TS+ccmax+cc] = rgb[rr*TS+ccmax+cc, c]
            
            # also, fill the image corners
            if rrmin > 0 and ccmin > 0:
                for rr in range(16):
                    for cc in range(16):
                        c = cfarray[rr, cc]
                        rgb[(rr)*TS+cc][c] = rgb[(32-rr)*TS+(32-cc)][c]
                        cfa[(rr)*TS+cc] = rgb[(rr)*TS+cc][c]
            
            if rrmax < rr1 and ccmax < cc1:
                for rr in range(16):
                    for cc in range(16):
                        c = cfarray[rr, cc]
                        rgb[(rrmax+rr)*TS+ccmax+cc][c] = (src[(winy+height-rr-2)][(winx+width-cc-2)])/65535
                        cfa[(rrmax+rr)*TS+ccmax+cc] = rgb[(rrmax+rr)*TS+ccmax+cc][c]
            
            if rrmin > 0 and ccmax < cc1:
                for rr in range(16):
                    for cc in range(16):
                        c = cfarray[rr, cc]
                        rgb[(rr)*TS+ccmax+cc][c] = (src[(winy+32-rr)][(winx+width-cc-2)])/65535
                        cfa[(rr)*TS+ccmax+cc] = rgb[(rr)*TS+ccmax+cc][c]
            
            if rrmax < rr1 and ccmin > 0:
                for rr in range(16):
                    for cc in range(16):
                        c = cfarray[rr, cc]
                        rgb[(rrmax+rr)*TS+cc][c] = (src[(winy+height-rr-2)][(winx+32-cc)])/65535
                        cfa[(rrmax+rr)*TS+cc] = rgb[(rrmax+rr)*TS+cc][c]
            
            # end of border fill

            for rr in range(1, rr1-1):
                for cc in range(1, cc1-1):
                    indx = rr*TS+cc
                    delh[indx] = abs(cfa[indx + 1] - cfa[indx - 1])
                    delv[indx] = abs(cfa[indx + v1] - cfa[indx - v1])
                    delhsq[indx] = SQR(delh[indx])
                    delvsq[indx] = SQR(delv[indx])
                    delp[indx] = abs(cfa[indx+p1]-cfa[indx-p1])
                    delm[indx] = abs(cfa[indx+m1]-cfa[indx-m1])

            for rr in range(2, rr1-2):
                for cc in range(2, cc1-2):
                    indx = rr*TS+cc
                    # vert directional averaging weights
                    dirwts[indx][0] = eps+delv[indx+v1]+delv[indx-v1]+delv[indx]
                    # horizontal weights
                    dirwts[indx][1] = eps+delh[indx+1]+delh[indx-1]+delh[indx]

                    if cfarray[rr, cc] & 1:
                        # for later use in diagonal interpolation
                        Dgrbpsq1[indx]=(SQR(cfa[indx]-cfa[indx-p1])+SQR(cfa[indx]-cfa[indx+p1]))
                        Dgrbmsq1[indx]=(SQR(cfa[indx]-cfa[indx-m1])+SQR(cfa[indx]-cfa[indx+m1]))
            
            for rr in range(4, rr1 - 4):
                for cc in range(4, cc1 - 4):
                    indx = rr*TS+cc
                    c = cfarray[rr, cc]
                    sgn = -1 if c & 1 else 1

                    # initialization of nyquist test
                    nyquist[indx]=0
                    # preparation for diag interp
                    rbint[indx]=0

                    # color ratios in each cardinal direction
                    cru = cfa[indx - v1] * (dirwts[indx - v2][0] + dirwts[indx][0]) / (dirwts[indx - v2][0] * (eps + cfa[indx]) + dirwts[indx][0] * (eps + cfa[indx - v2]))
                    crd = cfa[indx + v1] * (dirwts[indx + v2][0] + dirwts[indx][0]) / (dirwts[indx + v2][0] * (eps + cfa[indx]) + dirwts[indx][0] * (eps + cfa[indx + v2]))
                    crl = cfa[indx - 1] * (dirwts[indx - 2][1] + dirwts[indx][1]) / (dirwts[indx - 2][1] * (eps + cfa[indx]) + dirwts[indx][1] * (eps + cfa[indx - 2]))
                    crr = cfa[indx + 1] * (dirwts[indx + 2][1] + dirwts[indx][1]) / (dirwts[indx + 2][1] * (eps + cfa[indx]) + dirwts[indx][1] * (eps + cfa[indx + 2]))

                    # G interpolated in vert/hor directions using Hamilton-Adams method
                    guha = min(clip_pt, cfa[indx - v1] + 0.5 * (cfa[indx] - cfa[indx - v2]))
                    gdha = min(clip_pt, cfa[indx + v1] + 0.5 * (cfa[indx] - cfa[indx + v2]))
                    glha = min(clip_pt, cfa[indx - 1] + 0.5 * (cfa[indx] - cfa[indx - 2]))
                    grha = min(clip_pt, cfa[indx + 1] + 0.5 * (cfa[indx] - cfa[indx + 2]))

                    # G interpolated in vert/hor directions using adaptive ratios
                    guar = cfa[indx] * cru if abs(1-cru) < arthresh else guha
                    gdar = cfa[indx] * crd if abs(1-crd) < arthresh else gdha
                    glar = cfa[indx] * crl if abs(1-crl) < arthresh else glha
                    grar = cfa[indx] * crr if abs(1-crr) < arthresh else grha

                    # adaptive weights for vertical/horizontal directions
                    hwt = dirwts[indx - 1][1] / (dirwts[indx - 1][1] + dirwts[indx + 1][1])
                    vwt = dirwts[indx - v1][0] / (dirwts[indx + v1][0] + dirwts[indx - v1][0])

                    # interpolated G via adaptive weighTS of cardinal evaluations
                    Gintvar = vwt * gdar + (1-vwt) * guar
                    Ginthar = hwt * grar + (1-hwt) * glar
                    Gintvha = vwt * gdha + (1-vwt) * guha
                    Ginthha = hwt * grha + (1-hwt) * glha
                    # interpolated color differences
                    vcd[indx] = sgn * (Gintvar-cfa[indx])
                    hcd[indx] = sgn * (Ginthar-cfa[indx])
                    vcdalt[indx] = sgn * (Gintvha-cfa[indx])
                    hcdalt[indx] = sgn * (Ginthha-cfa[indx])

                    if cfa[indx] > 0.8 * clip_pt or Gintvha > 0.8 * clip_pt or Ginthha > 0.8 * clip_pt:
                        # use HA if highlighTS are (nearly) clipped
                        guar = guha
                        gdar = gdha
                        glar = glha
                        grar = grha
                        vcd[indx] = vcdalt[indx]
                        hcd[indx] = hcdalt[indx]

                    # differences of interpolations in opposite directions
                    dgintv[indx] = min((guha - gdha) ** 2, (guar - gdar) ** 2)
                    dginth[indx] = min((glha - grha) ** 2, (glar - grar) ** 2)
            
            for rr in range(4, rr1-4):
                for cc in range(4, cc1-4):
                    c = cfarray[rr, cc]

                    hcdvar = 3*(SQR(hcd[indx-2])+SQR(hcd[indx])+SQR(hcd[indx+2]))-SQR(hcd[indx-2]+hcd[indx]+hcd[indx+2])
                    hcdaltvar = 3*(SQR(hcdalt[indx-2])+SQR(hcdalt[indx])+SQR(hcdalt[indx+2]))-SQR(hcdalt[indx-2]+hcdalt[indx]+hcdalt[indx+2])
                    vcdvar = 3*(SQR(vcd[indx-v2])+SQR(vcd[indx])+SQR(vcd[indx+v2]))-SQR(vcd[indx-v2]+vcd[indx]+vcd[indx+v2])
                    vcdaltvar = 3*(SQR(vcdalt[indx-v2])+SQR(vcdalt[indx])+SQR(vcdalt[indx+v2]))-SQR(vcdalt[indx-v2]+vcdalt[indx]+vcdalt[indx+v2])

                    # choose the smallest variance; this yields a smoother interpolation
                    if hcdaltvar < hcdvar:
                        hcd[indx] = hcdalt[indx]
                    if vcdaltvar < vcdvar:
                        vcd[indx] = vcdalt[indx]

                    # bound the interpolation in regions of high saturation
                    # vertical and horizontal G interpolations
                    if c & 1: # G site
                        Ginth = -hcd[indx] + cfa[indx]
                        Gintv = -vcd[indx] + cfa[indx]

                        if hcd[indx] > 0:
                            if 3 * hcd[indx] > (Ginth + cfa[indx]):
                                hcd[indx] = -np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) + cfa[indx]
                            else:
                                hwt = 1 - 3 * hcd[indx] / (eps + Ginth + cfa[indx])
                                hcd[indx] = hwt * hcd[indx] + (1 - hwt) * (-np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) + cfa[indx])

                        if vcd[indx] > 0:
                            if 3 * vcd[indx] > (Gintv + cfa[indx]):
                                vcd[indx] = -np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) + cfa[indx]
                            else:
                                vwt = 1 - 3 * vcd[indx] / (eps + Gintv + cfa[indx])
                                vcd[indx] = vwt * vcd[indx] + (1 - vwt) * (-np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) + cfa[indx])
                            
                        if Ginth > clip_pt:
                            hcd[indx] = -np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) + cfa[indx]

                        if Gintv > clip_pt:
                            vcd[indx] = -np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) + cfa[indx]
                    
                    else: # R or B site

                        Ginth = hcd[indx] + cfa[indx]
                        Gintv = vcd[indx] + cfa[indx]

                        if hcd[indx] < 0:
                            if 3 * hcd[indx] < -(Ginth + cfa[indx]):
                                hcd[indx] = np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) - cfa[indx]
                            else:
                                hwt = 1 + 3 * hcd[indx] / (eps + Ginth + cfa[indx])
                                hcd[indx] = hwt * hcd[indx] + (1 - hwt) * (np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) - cfa[indx])

                        if vcd[indx] < 0:
                            if 3 * vcd[indx] < -(Gintv + cfa[indx]):
                                vcd[indx] = np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) - cfa[indx]
                            else:
                                vwt = 1 + 3 * vcd[indx] / (eps + Gintv + cfa[indx])
                                vcd[indx] = vwt * vcd[indx] + (1 - vwt) * (np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) - cfa[indx])

                        if Ginth > clip_pt:
                            hcd[indx] = np.median([Ginth, cfa[indx - 1], cfa[indx + 1]]) - cfa[indx]

                        if Gintv > clip_pt:
                            vcd[indx] = np.median([Gintv, cfa[indx - v1], cfa[indx + v1]]) - cfa[indx]

                    vcdsq[indx] = SQR(vcd[indx])
                    hcdsq[indx] = SQR(hcd[indx])
                    cddiffsq[indx] = SQR(vcd[indx]-hcd[indx])

            for rr in range(6, rr1-6):
                for cc in range(6+(cfarray[rr, 2]&1), cc1-6, 2):
                    indx = rr * TS + cc

                    # compute color difference variances in cardinal directions

                    Dgrbvvaru = 4*(vcdsq[indx]+vcdsq[indx-v1]+vcdsq[indx-v2]+vcdsq[indx-v3])-SQR(vcd[indx]+vcd[indx-v1]+vcd[indx-v2]+vcd[indx-v3])
                    Dgrbvvard = 4*(vcdsq[indx]+vcdsq[indx+v1]+vcdsq[indx+v2]+vcdsq[indx+v3])-SQR(vcd[indx]+vcd[indx+v1]+vcd[indx+v2]+vcd[indx+v3])
                    Dgrbhvarl = 4*(hcdsq[indx]+hcdsq[indx-1]+hcdsq[indx-2]+hcdsq[indx-3])-SQR(hcd[indx]+hcd[indx-1]+hcd[indx-2]+hcd[indx-3])
                    Dgrbhvarr = 4*(hcdsq[indx]+hcdsq[indx+1]+hcdsq[indx+2]+hcdsq[indx+3])-SQR(hcd[indx]+hcd[indx+1]+hcd[indx+2]+hcd[indx+3])
					
                    hwt = dirwts[indx-1][1]/(dirwts[indx-1][1]+dirwts[indx+1][1])
                    vwt = dirwts[indx-v1][0]/(dirwts[indx+v1][0]+dirwts[indx-v1][0])
					
                    vcdvar = epssq+vwt*Dgrbvvard+(1-vwt)*Dgrbvvaru
                    hcdvar = epssq+hwt*Dgrbhvarr+(1-hwt)*Dgrbhvarl

                    # compute fluctuations in up/down and left/right interpolations of colors
                    Dgrbvvaru = (dgintv[indx])+(dgintv[indx-v1])+(dgintv[indx-v2])
                    Dgrbvvard = (dgintv[indx])+(dgintv[indx+v1])+(dgintv[indx+v2])
                    Dgrbhvarl = (dginth[indx])+(dginth[indx-1])+(dginth[indx-2])
                    Dgrbhvarr = (dginth[indx])+(dginth[indx+1])+(dginth[indx+2])

                    vcdvar1 = epssq+vwt*Dgrbvvard+(1-vwt)*Dgrbvvaru
                    hcdvar1 = epssq+hwt*Dgrbhvarr+(1-hwt)*Dgrbhvarl

                    # determine adaptive weights for G interpolation
                    varwt=hcdvar/(vcdvar+hcdvar)
                    diffwt=hcdvar1/(vcdvar1+hcdvar1)

                    # if both agree on interpolation direction, choose the one with strongest directional discrimination;
                    # otherwise, choose the u/d and l/r difference fluctuation weights
                    if ((0.5 - varwt) * (0.5 - diffwt) > 0) and (abs(0.5 - diffwt) < abs(0.5 - varwt)):
                        hvwt[indx] = varwt
                    else:
                        hvwt[indx] = diffwt
            
            # Nyquist test
            for rr in range(6, rr1-6):
                for cc in range(6 + (cfarray[rr, 2]&1), cc1 - 6, 2):
                    indx = rr * TS + cc

                    # nyquist texture test: ask if difference of vcd compared to hcd is larger or smaller than RGGB gradients
                    nyqtest = (gaussodd[0]*cddiffsq[indx] + gaussodd[1]*(cddiffsq[indx-m1]+cddiffsq[indx+p1] + cddiffsq[indx-p1]+cddiffsq[indx+m1]) + gaussodd[2]*(cddiffsq[indx-v2]+cddiffsq[indx-2]+ cddiffsq[indx+2]+cddiffsq[indx+v2]) + gaussodd[3]*(cddiffsq[indx-m2]+cddiffsq[indx+p2] + cddiffsq[indx-p2]+cddiffsq[indx+m2]))

                    nyqtest -= nyqthresh*(gaussgrad[0]*(delhsq[indx]+delvsq[indx])+gaussgrad[1]*(delhsq[indx-v1]+delvsq[indx-v1]+delhsq[indx+1]+delvsq[indx+1] + delhsq[indx-1]+delvsq[indx-1]+delhsq[indx+v1]+delvsq[indx+v1])+ gaussgrad[2]*(delhsq[indx-m1]+delvsq[indx-m1]+delhsq[indx+p1]+delvsq[indx+p1]+ delhsq[indx-p1]+delvsq[indx-p1]+delhsq[indx+m1]+delvsq[indx+m1])+ gaussgrad[3]*(delhsq[indx-v2]+delvsq[indx-v2]+delhsq[indx-2]+delvsq[indx-2]+ delhsq[indx+2]+delvsq[indx+2]+delhsq[indx+v2]+delvsq[indx+v2])+ gaussgrad[4]*(delhsq[indx-2*TS-1]+delvsq[indx-2*TS-1]+delhsq[indx-2*TS+1]+delvsq[indx-2*TS+1]+ delhsq[indx-TS-2]+delvsq[indx-TS-2]+delhsq[indx-TS+2]+delvsq[indx-TS+2]+ delhsq[indx+TS-2]+delvsq[indx+TS-2]+delhsq[indx+TS+2]+delvsq[indx-TS+2]+ delhsq[indx+2*TS-1]+delvsq[indx+2*TS-1]+delhsq[indx+2*TS+1]+delvsq[indx+2*TS+1])+ gaussgrad[5]*(delhsq[indx-m2]+delvsq[indx-m2]+delhsq[indx+p2]+delvsq[indx+p2]+ delhsq[indx-p2]+delvsq[indx-p2]+delhsq[indx+m2]+delvsq[indx+m2]))

                    if nyqtest > 0:
                        # nyquist=1 for nyquist region
                        nyquist[indx] = 1
            
            for rr in range(8, rr1-8):
                for cc in range(8+(cfarray[rr,2]&1), cc1-8, 2):

                    areawt=(nyquist[indx-v2]+nyquist[indx-m1]+nyquist[indx+p1]+nyquist[indx-2]+nyquist[indx]+nyquist[indx+2]+nyquist[indx-p1]+nyquist[indx+m1]+nyquist[indx+v2])

                    # if most of your neighbors are named Nyquist, it's likely that you're one too
                    nyquist[indx] = 1 if areawt > 4 else 0

            # end of Nyquist test

            # in areas of Nyquist texture, do area interpolation
            for rr in range(8, rr1 - 8):
                for cc in range(8+(cfarray[rr,2]&1), cc1-8, 2):
                    indx = rr * TS + cc
                    if nyquist[indx]:
                        # area interpolation
                        sumh = sumv = sumsqh = sumsqv = areawt = 0
                        for i in range(-6, 7, 2):
                            for j in range(-6, 7, 2):
                                indx1 = (rr + i) * TS + cc + j
                                if nyquist[indx1]:
                                    sumh += cfa[indx1] - 0.5 * (cfa[indx1-1]+cfa[indx1+1])
                                    sumv += cfa[indx1] - 0.5 * (cfa[indx1-v1]+cfa[indx1+v1])
                                    sumsqh += 0.5 * (SQR(cfa[indx1]-cfa[indx1-1]) + SQR(cfa[indx1]-cfa[indx1+1]))
                                    sumsqv += 0.5 * (SQR(cfa[indx1]-cfa[indx1-v1]) + SQR(cfa[indx1]-cfa[indx1+v1]))
                                    areawt += 1

                        # horizontal and vertical color differences, and adaptive weight
                        hcdvar = epssq + max(0, areawt*sumsqh-sumh*sumh)
                        vcdvar = epssq + max(0, areawt*sumsqv-sumv*sumv)
                        hvwt[indx] = hcdvar / (vcdvar + hcdvar)
                    
                        # end of area interpolation
            
            # populate G at R/B sites
            for rr in range(8, rr1-8):
                for cc in range(8+(cfarray[rr,2]&1), cc1-8, 2):
                    indx = rr * TS + cc

                    # first ask if one gets more directional discrimination from nearby B/R sites
                    hvwtalt = 0.25 * (hvwt[indx-m1] + hvwt[indx+p1] + hvwt[indx-p1] + hvwt[indx+m1])
                    vo = abs(0.5 - hvwt[indx])
                    ve = abs(0.5 - hvwtalt)
                    # a better result was obtained from the neighbors
                    if vo < ve:
                        hvwt[indx>>1] = hvwtalt
                    # evaluate color differences
                    Dgrb[indx][0] = (hcd[indx]*(1-hvwt[indx]) + vcd[indx]*hvwt[indx])
                    # evaluate G
                    rgb[indx][1] = cfa[indx] + Dgrb[indx][0]
                    # local curvature in G (preparation for nyquist refinement step)
                    if nyquist[indx]:
                        Dgrbh2[indx] = SQR(rgb[indx][1] - 0.5*(rgb[indx-1][1]+rgb[indx+1][1]))
                        Dgrbv2[indx] = SQR(rgb[indx][1] - 0.5*(rgb[indx-v1][1]+rgb[indx+v1][1]))
                    else:
                        Dgrbh2[indx] = Dgrbv2[indx] = 0

            # end of standard interpolation


            # refine Nyquist areas using G curvatures
            for rr in range(8, rr1-8):
                for cc in range(8+(cfarray[rr,2]&1), cc1-8, 2):
                    indx = rr * TS + cc
                    if nyquist[indx]:
                        # local averages (over Nyquist pixels only) of G curvature squared 
                        gvarh = epssq + (gquinc[0]*Dgrbh2[indx]+gquinc[1]*(Dgrbh2[indx-m1]+Dgrbh2[indx+p1]+Dgrbh2[indx-p1]+Dgrbh2[indx+m1])+gquinc[2]*(Dgrbh2[indx-v2]+Dgrbh2[indx-2]+Dgrbh2[indx+2]+Dgrbh2[indx+v2])+gquinc[3]*(Dgrbh2[indx-m2]+Dgrbh2[indx+p2]+Dgrbh2[indx-p2]+Dgrbh2[indx+m2]))
                        gvarv = epssq + (gquinc[0]*Dgrbv2[indx]+gquinc[1]*(Dgrbv2[indx-m1]+Dgrbv2[indx+p1]+Dgrbv2[indx-p1]+Dgrbv2[indx+m1])+gquinc[2]*(Dgrbv2[indx-v2]+Dgrbv2[indx-2]+Dgrbv2[indx+2]+Dgrbv2[indx+v2])+gquinc[3]*(Dgrbv2[indx-m2]+Dgrbv2[indx+p2]+Dgrbv2[indx-p2]+Dgrbv2[indx+m2]))
                        # use the results as weights for refined G interpolation
                        Dgrb[indx][0] = (hcd[indx]*gvarv + vcd[indx]*gvarh)/(gvarv+gvarh)
                        rgb[indx][1] = cfa[indx] + Dgrb[indx][0]
            
            # diagonal interpolation correction
            for rr in range(8, rr1-8):
                for cc in range(8+(cfarray[rr,2]&1), cc1-8, 2):
                    indx = rr * TS + cc
                    rbvarp = epssq + (gausseven[0]*(Dgrbpsq1[indx-v1]+Dgrbpsq1[indx-1]+Dgrbpsq1[indx+1]+Dgrbpsq1[indx+v1]) + gausseven[1]*(Dgrbpsq1[indx-v2-1]+Dgrbpsq1[indx-v2+1]+Dgrbpsq1[indx-2-v1]+Dgrbpsq1[indx+2-v1]+ Dgrbpsq1[indx-2+v1]+Dgrbpsq1[indx+2+v1]+Dgrbpsq1[indx+v2-1]+Dgrbpsq1[indx+v2+1]))
                    rbvarm = epssq + (gausseven[0]*(Dgrbmsq1[indx-v1]+Dgrbmsq1[indx-1]+Dgrbmsq1[indx+1]+Dgrbmsq1[indx+v1]) + gausseven[1]*(Dgrbmsq1[indx-v2-1]+Dgrbmsq1[indx-v2+1]+Dgrbmsq1[indx-2-v1]+Dgrbmsq1[indx+2-v1]+ Dgrbmsq1[indx-2+v1]+Dgrbmsq1[indx+2+v1]+Dgrbmsq1[indx+v2-1]+Dgrbmsq1[indx+v2+1]))

                    # diagonal color ratios
                    crse=2*(cfa[indx+m1])/(eps+cfa[indx]+(cfa[indx+m2]))
                    crnw=2*(cfa[indx-m1])/(eps+cfa[indx]+(cfa[indx-m2]))
                    crne=2*(cfa[indx+p1])/(eps+cfa[indx]+(cfa[indx+p2]))
                    crsw=2*(cfa[indx-p1])/(eps+cfa[indx]+(cfa[indx-p2]))

                    # assign B/R at R/B sites
                    if abs(1 - crse) < arthresh:
                        rbse = cfa[indx] * crse
                    else:
                        rbse = cfa[indx + m1] + 0.5 * (cfa[indx] - cfa[indx + m2])

                    if abs(1 - crnw) < arthresh:
                        rbnw = (cfa[indx - m1]) + 0.5 *(cfa[indx] - cfa[indx - m2])

                    if abs(1 - crne) < arthresh:
                        rbne = cfa[indx] * crne
                    else:
                        rbne = (cfa[indx + p1]) + 0.5 * cfa[indx] - cfa[indx + p2]

                    if abs(1 - crsw) < arthresh:
                        rbsw = cfa[indx] * crsw
                    else:
                        rbsw = (cfa[indx - p1]) + 0.5 * (cfa[indx] - cfa[indx - p2])
                    
                    wtse= eps+delm[indx]+delm[indx+m1]+delm[indx+m2] # same as for wtu,wtd,wtl,wtr
                    wtnw= eps+delm[indx]+delm[indx-m1]+delm[indx-m2]
                    wtne= eps+delp[indx]+delp[indx+p1]+delp[indx+p2]
                    wtsw= eps+delp[indx]+delp[indx-p1]+delp[indx-p2]

                    rbm[indx] = (wtse*rbnw+wtnw*rbse)/(wtse+wtnw)
                    rbp[indx] = (wtne*rbsw+wtsw*rbne)/(wtne+wtsw)

                    pmwt[indx] = rbvarm/(rbvarp+rbvarm)

                    # bound the interpolation in regions of high saturation
                    if rbp[indx] < cfa[indx]:
                        if 2 * (rbp[indx]) < cfa[indx]:
                            rbp[indx] = np.median([rbp[indx] , cfa[indx - p1], cfa[indx + p1]])
                        else:
                            pwt = 2 * (cfa[indx] - rbp[indx]) / (eps + rbp[indx] + cfa[indx])
                            rbp[indx] = pwt * rbp[indx] + (1 - pwt) * np.median([rbp[indx], cfa[indx - p1], cfa[indx + p1]])

                    if rbm[indx] < cfa[indx]:
                        if 2 * (rbm[indx]) < cfa[indx]:
                            rbm[indx] = np.median([rbm[indx] , cfa[indx - m1], cfa[indx + m1]])
                        else:
                            mwt = 2 * (cfa[indx] - rbm[indx]) / (eps + rbm[indx] + cfa[indx])
                            rbm[indx] = mwt * rbm[indx] + (1 - mwt) * np.median([rbm[indx], cfa[indx - m1], cfa[indx + m1]])

                    if rbp[indx] > clip_pt:
                        rbp[indx] = np.median([rbp[indx], cfa[indx - p1], cfa[indx + p1]])

                    if rbm[indx] > clip_pt:
                        rbm[indx] = np.median([rbm[indx], cfa[indx - m1], cfa[indx + m1]])

            for rr in range(10, rr1-10):
                for cc in range(10 + (cfarray[rr, 2]&1), cc1-10, 2):
                    indx = rr * TS + cc
                    
                    # first ask if one geTS more directional discrimination from nearby B/R sites
                    pmwtalt = 0.25*(pmwt[indx-m1]+pmwt[indx+p1]+pmwt[indx-p1]+pmwt[indx+m1])
                    vo = abs(0.5-pmwt[indx])
                    ve = abs(0.5-pmwtalt)
                    if vo < ve:
                        pmwt[indx] = pmwtalt
                    rbint[indx] = 0.5*(cfa[indx] + rbm[indx]*(1-pmwt[indx]) + rbp[indx]*pmwt[indx])

            for rr in range(12, rr1 - 12):
                for cc in range(12 + (cfarray[rr, 2]&1), cc1 - 12, 2):
                    indx = rr * TS + cc
                    if abs(0.5 - pmwt[indx]) < abs(0.5 - hvwt[indx]):
                        continue
                    
                    # now interpolate G vertically/horizontally using R+B values
                    # unfortunately, since G interpolation cannot be done diagonally this may lead to colour shifts
                    # colour ratios for G interpolation
                    cru = cfa[indx-v1]*2/(eps+rbint[indx]+rbint[indx-v2])
                    crd = cfa[indx+v1]*2/(eps+rbint[indx]+rbint[indx+v2])
                    crl = cfa[indx-1]*2/(eps+rbint[indx]+rbint[indx-2])
                    crr = cfa[indx+1]*2/(eps+rbint[indx]+rbint[indx+2])

                    # interpolated G via adaptive ratios or Hamilton-Adams in each cardinal direction
                    if abs(1 - cru) < arthresh:
                        gu = rbint[indx] * cru
                    else:
                        gu = cfa[indx - v1] + 0.5 * (rbint[indx] - rbint[(indx - v1)])

                    if abs(1 - crd) < arthresh:
                        gd = rbint[indx] * crd
                    else:
                        gd = cfa[indx + v1] + 0.5 * (rbint[indx] - rbint[(indx + v1)])

                    if abs(1 - crl) < arthresh:
                        gl = rbint[indx] * crl
                    else:
                        gl = cfa[indx - 1] + 0.5 * (rbint[indx] - rbint[(indx - 1)])

                    if abs(1 - crr) < arthresh:
                        gr = rbint[indx] * crr
                    else:
                        gr = cfa[indx + 1] + 0.5 * (rbint[indx] - rbint[(indx + 1)])

                    # interpolated G via adaptive weighTS of cardinal evaluations
                    Gintv = (dirwts[indx - v1][0] * gd + dirwts[indx + v1][0] * gu) / (dirwts[indx + v1][0] + dirwts[indx - v1][0])
                    Ginth = (dirwts[indx - 1][1] * gr + dirwts[indx + 1][1] * gl) / (dirwts[indx - 1][1] + dirwts[indx + 1][1])

                    # bound the interpolation in regions of high saturation
                    if Gintv < rbint[indx]:
                        if (2 * Gintv < rbint[indx]):
                            Gintv = np.median([Gintv , cfa[indx - v1], cfa[indx + v1]])
                        else:
                            vwt = 2 * (rbint[indx] - Gintv) / (eps + Gintv + rbint[indx])
                            Gintv = vwt * Gintv + (1 - vwt) * np.median([Gintv, cfa[indx - v1], cfa[indx + v1]])

                    if Ginth < rbint[indx]:
                        if 2 * Ginth < rbint[indx]:
                            Ginth = np.median([Ginth , cfa[indx - 1], cfa[indx + 1]])
                        else:
                            hwt = 2 * (rbint[indx] - Ginth) / (eps + Ginth + rbint[indx])
                            Ginth = hwt * Ginth + (1 - hwt) * np.median([Ginth, cfa[indx - 1], cfa[indx + 1]])
                    
                    if Ginth > clip_pt:
                        Ginth = np.median([Ginth, cfa[indx - 1], cfa[indx + 1]])

                    if Gintv > clip_pt:
                        Gintv = np.median([Gintv, cfa[indx - v1], cfa[indx + v1]])
                    
                    rgb[indx][1] = Ginth*(1-hvwt[indx]) + Gintv*hvwt[indx]
                    Dgrb[indx][0] = rgb[indx][1]-cfa[indx]

            # end of diagonal interpolation correction

            # fancy chrominance interpolation
            # (ey,ex) is location of R site
            for rr in range(13-ey, rr1-12, 2):
                for cc in range(13-ex, cc1-12, 2):
                    indx = rr*TS+cc
                    Dgrb[indx][1]=Dgrb[indx][0] # split out G-B from G-R
                    Dgrb[indx][0]=0

            for rr in range(12, rr1-12):
                c = int(1- cfarray[rr, 12+(cfarray[rr,2]&1)]/2)
                for cc in range(12+(cfarray[rr,2]&1), cc1-12, 2):
                    indx = rr * TS + cc
                    wtnw=1/(eps+abs(Dgrb[indx-m1][c]-Dgrb[indx+m1][c])+abs(Dgrb[indx-m1][c]-Dgrb[indx-m3][c])+abs(Dgrb[indx+m1][c]-Dgrb[indx-m3][c]))
                    wtne=1/(eps+abs(Dgrb[indx+p1][c]-Dgrb[indx-p1][c])+abs(Dgrb[indx+p1][c]-Dgrb[indx+p3][c])+abs(Dgrb[indx-p1][c]-Dgrb[indx+p3][c]))
                    wtsw=1/(eps+abs(Dgrb[indx-p1][c]-Dgrb[indx+p1][c])+abs(Dgrb[indx-p1][c]-Dgrb[indx+m3][c])+abs(Dgrb[indx+p1][c]-Dgrb[indx-p3][c]))
                    wtse=1/(eps+abs(Dgrb[indx+m1][c]-Dgrb[indx-m1][c])+abs(Dgrb[indx+m1][c]-Dgrb[indx-p3][c])+abs(Dgrb[indx-m1][c]-Dgrb[indx+m3][c]))
                    
                    Dgrb[indx][c]=(wtnw*(1.325*Dgrb[indx-m1][c]-0.175*Dgrb[indx-m3][c]-0.075*Dgrb[indx-m1-2][c]-0.075*Dgrb[indx-m1-v2][c] )+ wtne*(1.325*Dgrb[indx+p1][c]-0.175*Dgrb[indx+p3][c]-0.075*Dgrb[indx+p1+2][c]-0.075*Dgrb[indx+p1+v2][c] )+ wtsw*(1.325*Dgrb[indx-p1][c]-0.175*Dgrb[indx-p3][c]-0.075*Dgrb[indx-p1-2][c]-0.075*Dgrb[indx-p1-v2][c] )+ wtse*(1.325*Dgrb[indx+m1][c]-0.175*Dgrb[indx+m3][c]-0.075*Dgrb[indx+m1+2][c]-0.075*Dgrb[indx+m1+v2][c] ))/(wtnw+wtne+wtsw+wtse)

            for rr in range(12, rr1-12):
                # c = int(cfarray[rr, 12+(cfarray[rr,1]&1)+1]/2)
                for cc in range(12+(cfarray[rr,1]&1), cc1-12, 2):
                    for c in range(2):
                        Dgrb[indx][c]=((hvwt[indx-v1])*Dgrb[indx-v1][c]+(1-hvwt[indx+1])*Dgrb[indx+1][c]+(1-hvwt[indx-1])*Dgrb[indx-1][c]+(hvwt[indx+v1])*Dgrb[indx+v1][c])/((hvwt[indx-v1])+(1-hvwt[indx+1])+(1-hvwt[indx-1])+(hvwt[indx+v1]))

            for rr in range(12, rr1-12):
                for cc in range(12, cc1-12):
                    indx = rr * TS + cc
                    rgb[indx][0]=(rgb[indx][1]-Dgrb[indx][0])
                    rgb[indx][2]=(rgb[indx][1]-Dgrb[indx][1])

            
            # copy smoothed results back to image matrix
            for rr in range(16, rr1-16):
                row = rr + top
                for cc in range(16, cc1-16):
                    col = cc + left
                    indx = row * width + col

                    for c in range(3):
                        image[row, col, c] = int(rgb[rr*TS+cc, c] * 65535 + 0.5)
    
            # end of main loop
    return image








