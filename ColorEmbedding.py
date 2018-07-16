import cv2
import pywt
import numpy as np
import Simulation as sm
import Transformations as tr


def IncorporateTexture(img, pure, K):
    # Transforming image to double type
    img = np.float32(img)

    # Changing domain to YCrCb
    img = tr.RGB2YCBCR(img)
    Y, Cr, Cb = cv2.split(img)

    # Wavelet Transformation in 2 levels
    (Sl, (Sh1, Sv1, Sd1), (Sh2, Sv2, Sd2)) = pywt.wavedec2(Y, 'db1', level=2)

    # Resizing layers
    ReducedCb = cv2.resize(
        Cb, (Sd2.shape[1], Sd2.shape[0]), interpolation=cv2.INTER_AREA)
    ReducedCr = cv2.resize(
        Cr, (Sv2.shape[1], Sv2.shape[0]), interpolation=cv2.INTER_AREA)

    # Building Cb+, Cb-, Cr+, Cr-
    CbPlus = np.zeros((ReducedCb.shape[0], ReducedCb.shape[1]))
    CbMinus = np.zeros((ReducedCb.shape[0], ReducedCb.shape[1]))

    # Updating values of Cb+ and Cb-
    for i in range(0, ReducedCb.shape[0]):
        for j in range(0, ReducedCb.shape[1]):
            if ReducedCb[i, j] < 0:
                CbPlus[i, j] = 0
                CbMinus[i, j] = ReducedCb[i, j]
            elif ReducedCb[i, j] > 0:
                CbMinus[i, j] = 0
                CbPlus[i, j] = ReducedCb[i, j]

    CrPlus = np.zeros((ReducedCr.shape[0], ReducedCr.shape[1]))
    CrMinus = np.zeros((ReducedCr.shape[0], ReducedCr.shape[1]))

    # Updating values of Cr+ and Cr-
    for i in range(0, ReducedCr.shape[0]):
        for j in range(0, ReducedCr.shape[1]):
            if ReducedCr[i, j] < 0:
                CrPlus[i, j] = 0
                CrMinus[i, j] = ReducedCr[i, j]
            elif ReducedCr[i, j] > 0:
                CrMinus[i, j] = 0
                CrPlus[i, j] = ReducedCr[i, j]

    # Resizing Cb- to 1/4 of original size
    ReducedCbMinus = cv2.resize(
        CbMinus, (Sd1.shape[1], Sd1.shape[0]), interpolation=cv2.INTER_AREA)

    # Changes
    # Sd1 = ReducedCbMinus
    # Sh2 = CrPlus
    # Sv2 = CbPlus
    # Sd2 = CrMinus

    # Inverse Wavelet Transformation
    NewYSecondTry = (pywt.waverec2(
        (Sl, (Sh1, Sv1, ReducedCbMinus), (CrPlus, CbPlus, CrMinus)), 'db1'))

    # Saving results
    if pure == 0:
        # Pure result
        return NewYSecondTry
    elif pure == 1:
        # Result with simulation
        NewYSecondTry = sm.SimulateRealWorld(NewYSecondTry, K)
        return NewYSecondTry
