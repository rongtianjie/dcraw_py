import numpy as np
from scipy.ndimage import convolve
import rawpy
import cv2

iterations = 2


def labf(t):
    # 本算法中传入的 t 是一个二维矩阵
    d = t ** (1 / 3)
    index = np.where(t <= 0.008856)  # 0.008856 约等于 (6/29) 的三次方
    d[index] = 7.787 * t[index] + 4 / 29  # 7.787 约等于 (1/3) * (29/6) * (29/6)
    return d

def rgb2lab(X):
    a = np.array([
        [3.40479, -1.537150, -0.498535],
        [-0.969256, 1.875992, 0.041556],
        [0.055648, -0.204043, 1.057311]])
    # ai = np.array([
    #     [0.38627512, 0.33488427, 0.1689713],
    #     [0.19917304, 0.70345694, 0.06626421],
    #     [0.01810671, 0.11812969, 0.94969014]])  # 该矩阵和下面算出来的是一样的
    ai = np.linalg.inv(a)  # 求矩阵a的逆矩阵

    h, w, c = X.shape  # X是含有RGB三个分量的数据
    R = X[:, :, 0]
    G = X[:, :, 1]
    B = X[:, :, 2]
    planed_R = R.flatten()  # 将二维矩阵转成1维矩阵
    planed_G = G.flatten()
    planed_B = B.flatten()
    planed_image = np.zeros((c, h * w))  # 注意这里 planed_B 是一个二维数组
    planed_image[0, :] = planed_R  # 将 planed_R 赋值给 planed_image 的第0行
    planed_image[1, :] = planed_G
    planed_image[2, :] = planed_B
    planed_lab = np.dot(ai, planed_image)  # 相当于两个矩阵相乘 将rgb空间转到xyz空间
    planed_1 = planed_lab[0, :]
    planed_2 = planed_lab[1, :]
    planed_3 = planed_lab[2, :]
    L1 = np.reshape(planed_1, (h, w))
    L2 = np.reshape(planed_2, (h, w))
    L3 = np.reshape(planed_3, (h, w))
    result_lab = np.zeros((h, w, c))
    # color  space conversion  into LAB
    result_lab[:, :, 0] = 116 * labf(L2 / 255) - 16
    result_lab[:, :, 1] = 500 * (labf(L1 / 255) - labf(L2 / 255))
    result_lab[:, :, 2] = 200 * (labf(L2 / 255) - labf(L3 / 255))

    return result_lab

def masks_Bayer(im, pattern):
    w, h = im.shape
    R = np.zeros((w, h))
    GR = np.zeros((w, h))
    GB = np.zeros((w, h))
    B = np.zeros((w, h))

    # 将对应位置的元素取出来,因为懒所以没有用效率最高的方法,大家可以自己去实现
    if pattern == "RGGB":
        R[::2, ::2] = 1
        GR[::2, 1::2] = 1
        GB[1::2, ::2] = 1
        B[1::2, 1::2] = 1
    elif pattern == "GRBG":
        GR[::2, ::2] = 1
        R[::2, 1::2] = 1
        B[1::2, ::2] = 1
        GB[1::2, 1::2] = 1
    elif pattern == "GBRG":
        GB[::2, ::2] = 1
        B[::2, 1::2] = 1
        R[1::2, ::2] = 1
        GR[1::2, 1::2] = 1
    elif pattern == "BGGR":
        B[::2, ::2] = 1
        GB[::2, 1::2] = 1
        GR[1::2, ::2] = 1
        R[1::2, 1::2] = 1
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    R_m = R
    G_m = GB + GR
    B_m = B
    return R_m, G_m, B_m

def AH_gradient(img, pattern):
    X = img
    Rm, Gm, Bm = masks_Bayer(img, pattern)
    # green
    Hg1 = np.array([0, 1, 0, -1, 0])
    Hg2 = np.array([-1, 0, 2, 0, -1])

    Hg1 = Hg1.reshape(1, -1)
    Hg2 = Hg2.reshape(1, -1)
    # 如果当前像素是绿色就用绿色梯度，不是绿色就用颜色加上绿色梯度
    Ga = (Rm + Bm) * (np.abs(convolve(X, Hg1, mode = 'constant')) + np.abs(convolve(X, Hg2, mode = 'constant')))

    return Ga

# 计算梯度
def AH_gradientX(img, pattern):
    Ga = AH_gradient(img, pattern)

    return Ga


def AH_gradientY(img, pattern):  # 计算y方向上的梯度相当于把图像进行了翻转，pattern就变了。
    if pattern == "RGGB":
        new_pattern = "RGGB"
    elif pattern == "GRBG":
        new_pattern = "GBRG"
    elif pattern == "GBRG":
        new_pattern = "GRBG"
    elif pattern == "BGGR":
        new_pattern = "BGGR"
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    new_img = img.T
    Ga = AH_gradient(new_img, new_pattern)
    new_Ga = Ga.T
    return new_Ga

# 插值函数
def AH_interpolate(img, pattern, gamma, max_value):
    X = img
    Rm, Gm, Bm = masks_Bayer(img, pattern)  # 得到rgb三个分量的模板

    # green
    Hg1 = np.array([0, 1 / 2, 0, 1 / 2, 0])
    Hg2 = np.array([-1 / 4, 0, 1 / 2, 0, -1 / 4])
    Hg = Hg1 + Hg2 * gamma  # shape 为 (5,) 矩阵Hg参考公众号的论文
    Hg = Hg.reshape(1, -1)  # shape 为 (1,5)
    G = Gm * X + (Rm + Bm) * convolve(X, Hg, mode = 'constant')  # 得到所有的G

    # red / blue
    Hr = [[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]]
    R = G + convolve(Rm * (X - G), Hr, mode = 'constant')
    B = G + convolve(Bm * (X - G), Hr, mode = 'constant')

    R = np.clip(R, 0, max_value)
    G = np.clip(G, 0, max_value)
    B = np.clip(B, 0, max_value)
    return R, G, B


# X方向进行插值
def AH_interpolateX(img, pattern, gamma, max_value):
    h, w = img.shape
    Y = np.zeros((h, w, 3))
    R, G, B = AH_interpolate(img, pattern, gamma, max_value)
    Y[:, :, 0] = R
    Y[:, :, 1] = G
    Y[:, :, 2] = B
    return Y


# Y方向进行插值
def AH_interpolateY(img, pattern, gamma, max_value):
    h, w = img.shape
    Y = np.zeros((h, w, 3))
    if pattern == "RGGB":
        new_pattern = "RGGB"
    elif pattern == "GRBG":
        new_pattern = "GBRG"
    elif pattern == "GBRG":
        new_pattern = "GRBG"
    elif pattern == "BGGR":
        new_pattern = "BGGR"
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    new_img = img.T
    R, G, B = AH_interpolate(new_img, new_pattern, gamma, max_value)
    Y[:, :, 0] = R.T
    Y[:, :, 1] = G.T
    Y[:, :, 2] = B.T
    return Y

def MNballset(delta):
    index = delta
    H = np.zeros((index * 2 + 1, index * 2 + 1, (index * 2 + 1) ** 2))  # initialize

    k = 0
    for i in range(-index, index + 1):
        for j in range(-index, index + 1):
            # if np.linalg.norm([i, j]) <= delta:
            if (i ** 2 + j ** 2) <= (delta ** 2):  # 上面的if语句和这句意思是一样的。
                H[index + i, index + j, k] = 1  # included
                k = k + 1
    H = H[:, :, 0:k]  # 截取最后一维0-k的元素重新赋值给H，相当于调整H的大小
    # 最终H得到如下所示的矩阵：
    # 当 delta 为1时，得到的三维矩阵里每一个二维矩阵都是下面这个矩阵的，一个位置上的元素为1，其余均为0，
    # H的所有二维矩阵加起来等于下面这个矩阵。
    # [[0. 1. 0.]
    #  [1. 1. 1.]
    #  [0. 1. 0.]]

    # 当 delta 为2时。得到的三维矩阵里每一个二维矩阵都是下面这个矩阵的，一个位置上的元素为1，其余均为0，
    # H的所有二维矩阵加起来等于下面这个矩阵。
    # [[0. 0. 1. 0. 0.]
    #  [0. 1. 1. 1. 0.]
    #  [1. 1. 1. 1. 1.]
    #  [0. 1. 1. 1. 0.]
    #  [0. 0. 1. 0. 0.]]
    # 从 delta=1 和delta=2可以看出，该函数的目的就是为了得到一个对角线长为2delta，中心点在矩阵中心的一个菱形。
    # 菱形上的点都为1，其他地方为0，总共为1的点的个数为 2*(delta ** 2) + 2*delta + 1
    # 再次强调一下，三维矩阵里的二维矩阵，只有一个元素为1，是所有的二维矩阵相加。才构成上面那个矩阵的。
    return H


# 计算得到 delta L 和 delta C 和imatest 的计算方法相似。
def MNparamA(YxLAB, YyLAB):
    X = YxLAB
    Y = YyLAB
    kernel_H1 = np.array([1, -1, 0])  # shape:(3,)
    kernel_H1 = kernel_H1.reshape(1, -1)  # shape:(1, 3) 相当于把一个一维矩阵转成一个二维矩阵
    kernel_H2 = np.array([0, -1, 1])
    kernel_H2 = kernel_H2.reshape(1, -1)
    kernel_V1 = kernel_H1.reshape(1, -1).T  # shape:(3, 1) 相当于把一个一维矩阵顺时针转90度。
    kernel_V2 = kernel_H2.reshape(1, -1).T

    eLM1 = np.maximum(np.abs(convolve(X[:, :, 0], kernel_H1, mode = 'constant')),
                      np.abs(convolve(X[:, :, 0], kernel_H2, mode = 'constant')))
    eLM2 = np.maximum(np.abs(convolve(Y[:, :, 0], kernel_V1, mode = 'constant')),
                      np.abs(convolve(Y[:, :, 0], kernel_V2, mode = 'constant')))
    eL = np.minimum(eLM1, eLM2)
    eCx = np.maximum(
        convolve(X[:, :, 1], kernel_H1, mode = 'constant') ** 2 + convolve(X[:, :, 2], kernel_H1, mode = 'constant') ** 2,
        convolve(X[:, :, 1], kernel_H2, mode = 'constant') ** 2 + convolve(X[:, :, 2], kernel_H2, mode = 'constant') ** 2)
    eCy = np.maximum(
        convolve(Y[:, :, 1], kernel_V2, mode = 'constant') ** 2 + convolve(Y[:, :, 2], kernel_V2, mode = 'constant') ** 2,
        convolve(Y[:, :, 1], kernel_V1, mode = 'constant') ** 2 + convolve(Y[:, :, 2], kernel_V1, mode = 'constant') ** 2)
    eC = np.minimum(eCx, eCy)
    eL = eL  # 相当于imatest计算 delta L
    eC = eC ** 0.5  # 相当于imatest计算 delta C  (delta L) ** 2 + (delta C) ** 2 = (delta E) ** 2
    return eL, eC


# 计算相似度ƒ 在像素周边找到和其类似像素的数量
def MNhomogeneity(LAB_image, delta, epsilonL, epsilonC):
    H = MNballset(delta)
    # 这里是否可以改进 根据 MNballset 最后的注释，将 2*(delta ** 2) + 2*delta + 1 个二维矩阵合并成一个二维矩阵
    X = LAB_image
    epsilonC_sq = epsilonC ** 2
    h, w, c = LAB_image.shape
    K = np.zeros((h, w))
    kh, kw, kc = H.shape
    print(H.shape, X.shape)  # (5, 5, 13) (788, 532, 3)
    # 注意浮点数精度可能会有影响
    for i in range(kc):
        L = np.abs(convolve(X[:, :, 0], H[:, :, i], mode = 'constant') - X[:, :, 0]) <= epsilonL  # level set
        C = ((convolve(X[:, :, 1], H[:, :, i], mode = 'constant') - X[:, :, 1]) ** 2 + (
                    convolve(X[:, :, 2], H[:, :, i], mode = 'constant') - X[:, :, 2]) ** 2) <= epsilonC_sq  # color set
        U = C & L  # metric neighborhood 度量邻域
        K = K + U  # homogeneity 同质性
        # print(L.shape, C.shape, U.shape, K.shape)  # shape 都是 (788, 532)
        # L C U 里的元素只有 true 和 false
        # L 和 C 同时为true，则认为该点与周围的点相似。所以K中某一个元素的值，表示该点与周围多少个点相似。
    # print("K", K)
    return K

# 去artifact
def MNartifact(R, G, B, iterations):
    h, w = R.shape
    Rt = np.zeros((h, w, 8))
    Bt = np.zeros((h, w, 8))
    Grt = np.zeros((h, w, 4))
    Gbt = np.zeros((h, w, 4))
    kernel_1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    kernel_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    kernel_3 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    kernel_4 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    kernel_5 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    kernel_7 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    kernel_8 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    kernel_9 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

    for i in range(iterations):
        Rt[:, :, 0] = convolve(R - G, kernel_1, mode = 'constant')
        Rt[:, :, 1] = convolve(R - G, kernel_2, mode = 'constant')
        Rt[:, :, 2] = convolve(R - G, kernel_3, mode = 'constant')
        Rt[:, :, 3] = convolve(R - G, kernel_4, mode = 'constant')
        Rt[:, :, 4] = convolve(R - G, kernel_6, mode = 'constant')
        Rt[:, :, 5] = convolve(R - G, kernel_7, mode = 'constant')
        Rt[:, :, 6] = convolve(R - G, kernel_8, mode = 'constant')
        Rt[:, :, 7] = convolve(R - G, kernel_9, mode = 'constant')

        Rm = np.median(Rt, axis=2)
        R = G + Rm

        Bt[:, :, 0] = convolve(B - G, kernel_1, mode = 'constant')
        Bt[:, :, 1] = convolve(B - G, kernel_2, mode = 'constant')
        Bt[:, :, 2] = convolve(B - G, kernel_3, mode = 'constant')
        Bt[:, :, 3] = convolve(B - G, kernel_4, mode = 'constant')
        Bt[:, :, 4] = convolve(B - G, kernel_6, mode = 'constant')
        Bt[:, :, 5] = convolve(B - G, kernel_7, mode = 'constant')
        Bt[:, :, 6] = convolve(B - G, kernel_8, mode = 'constant')
        Bt[:, :, 7] = convolve(B - G, kernel_9, mode = 'constant')

        Bm = np.median(Bt, axis=2)
        B = G + Bm

        Grt[:, :, 0] = convolve(G - R, kernel_2, mode = 'constant')
        Grt[:, :, 1] = convolve(G - R, kernel_4, mode = 'constant')
        Grt[:, :, 2] = convolve(G - R, kernel_6, mode = 'constant')
        Grt[:, :, 3] = convolve(G - R, kernel_8, mode = 'constant')
        Grm = np.median(Grt, axis=2)
        Gr = R + Grm

        Gbt[:, :, 0] = convolve(G - B, kernel_2, mode = 'constant')
        Gbt[:, :, 1] = convolve(G - B, kernel_4, mode = 'constant')
        Gbt[:, :, 2] = convolve(G - B, kernel_6, mode = 'constant')
        Gbt[:, :, 3] = convolve(G - B, kernel_8, mode = 'constant')
        Gbm = np.median(Gbt, axis=2)
        Gb = B + Gbm
        G = (Gr + Gb) / 2
    return R, G, B

# adams hamilton
def AH_demosaic(img, pattern, gamma=1):
    print("AH demosaic start")
    if str(img.dtype) == 'uint16':
        max_value = 65535
    else:
        max_value = 255
    imgh, imgw = img.shape
    imgs = 10  # 向外延申10个像素不改变 pattern。

    # 扩展大小
    f = np.pad(img, ((imgs, imgs), (imgs, imgs)), 'reflect')
    # X,Y方向插值
    print('interpolate')
    Yx = AH_interpolateX(f, pattern, gamma, max_value)  # 对x方向进行插值,得到的是含有RGB三个分量的数据
    Yy = AH_interpolateY(f, pattern, gamma, max_value)  # 对y方向进行插值,得到的是含有RGB三个分量的数据
    print('gradient')
    Hx = AH_gradientX(f, pattern)  # 这边计算的是梯度，梯度越小，相似度就越大
    Hy = AH_gradientY(f, pattern)
    # set output to Yy if Hy <= Hx
    index = np.where(Hy <= Hx)
    R = Yx[:, :, 0]
    G = Yx[:, :, 1]
    B = Yx[:, :, 2]
    Ry = Yy[:, :, 0]
    Gy = Yy[:, :, 1]
    By = Yy[:, :, 2]
    Rs = R
    Gs = G
    Bs = B
    Rs[index] = Ry[index]
    Gs[index] = Gy[index]
    Bs[index] = By[index]
    h, w = Rs.shape
    Y = np.zeros((h, w, 3))
    Y[:, :, 0] = Rs
    Y[:, :, 1] = Gs
    Y[:, :, 2] = Bs
    # 调整size和值的范畴
    # Y = np.clip(Y, 0, max_value)
    resultY = Y[imgs:imgs + imgh, imgs:imgs + imgw, :]
    return resultY.astype(np.uint16)

def ahd_demosaic(bayer, pattern, delta = 2, gamma = 1):
    height, width = bayer.shape
    if str(bayer.dtype) == 'uint16':
        maxvalue = 65535
    else:
        maxvalue = 255
    p = 10
    f = np.pad(bayer, ((p,p),(p,p)), 'reflect')
    Yx = AH_interpolateX(f, pattern, gamma, maxvalue)
    Yy = AH_interpolateY(f, pattern, gamma, maxvalue)

    YxLAB = rgb2lab(Yx)
    YyLAB = rgb2lab(Yy)

    epsilonL, epsilonC = MNparamA(YxLAB, YyLAB)
    Hx = MNhomogeneity(YxLAB, delta, epsilonL, epsilonC)
    Hy = MNhomogeneity(YyLAB, delta, epsilonL, epsilonC)

    f_kernel = np.ones((3, 3))
    Hx = convolve(Hx, f_kernel, mode = 'constant')
    Hy = convolve(Hy, f_kernel, mode = 'constant')

    R = Yx[:, :, 0]
    G = Yx[:, :, 1]
    B = Yx[:, :, 2]
    Ry = Yy[:, :, 0]
    Gy = Yy[:, :, 1]
    By = Yy[:, :, 2]

    bigger_index = np.where(Hy >= Hx)
    Rs = R
    Gs = G
    Bs = B
    Rs[bigger_index] = Ry[bigger_index]
    Gs[bigger_index] = Gy[bigger_index]
    Bs[bigger_index] = By[bigger_index]
    h, w = Rs.shape
    YT = np.zeros((h, w, 3))
    YT[:, :, 0] = Rs
    YT[:, :, 1] = Gs
    YT[:, :, 2] = Bs
    # 去掉artifact
    Rsa, Gsa, Bsa = MNartifact(Rs, Gs, Bs, iterations)  # find
    Y = np.zeros((h, w, 3))
    Y[:, :, 0] = Rsa
    Y[:, :, 1] = Gsa
    Y[:, :, 2] = Bsa

    # 调整size和值的范畴
    Y = np.clip(Y, 0, maxvalue)
    resultY = Y[p:height + p, p:width + p, :]
    return resultY.astype(np.uint16)

def CLIP(src):
    rslt = src.copy()
    rslt[rslt>65536] = 65535
    rslt[rslt<0] = 0
    return rslt

if __name__ == '__main__':
    raw = rawpy.imread('D:/images/DSCF0145.RAF')

    rslt = raw.raw_image[raw.sizes.top_margin:raw.sizes.top_margin+raw.sizes.iheight, raw.sizes.left_margin:raw.sizes.left_margin+raw.sizes.iwidth].astype(np.int32)

    for i, bl in enumerate(raw.black_level_per_channel):
        rslt[raw.raw_colors[raw.sizes.top_margin:raw.sizes.top_margin+raw.sizes.iheight, raw.sizes.left_margin:raw.sizes.left_margin+raw.sizes.iwidth] == i] -= bl

    wb = raw.camera_whitebalance
    wb_coeff = np.asarray(wb[:3]) / max(wb[:3])
    wb_coeff = np.append(wb_coeff,wb_coeff[1])

    if raw.camera_white_level_per_channel is None:
        white_level = [65535] * 4
    else:
        white_level = raw.camera_white_level_per_channel
    
    white_level = np.array(white_level) - np.array(raw.black_level_per_channel)

    scale_coeff = wb_coeff * 65535 / white_level

    scale_matrix = np.empty([raw.sizes.iheight, raw.sizes.iwidth], dtype=np.float32)

    for i, scale_co in  enumerate(scale_coeff):
        scale_matrix[raw.raw_colors[raw.sizes.top_margin:raw.sizes.top_margin+raw.sizes.iheight, raw.sizes.left_margin:raw.sizes.left_margin+raw.sizes.iwidth] == i] = scale_co

    rslt = CLIP(rslt * scale_matrix).astype(np.uint16)

    # image = AH_demosaic(rslt, 'RGGB')
    image = ahd_demosaic(rslt, 'RGGB')

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.png", image)

