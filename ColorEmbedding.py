import cv2
import pywt
import numpy as np
import Simulation as sm
import Transformations as tr


# Função que calcula Cb/Cr-menos e mais
def plus_minus(img):
    height, width = img.shape

    plus = np.zeros((height, width), np.float32)
    minus = np.zeros((height, width), np.float32)
    # separa a matriz em suas partes negativa e positiva
    minus = (img - np.abs(img)) / 2
    plus = (img + np.abs(img)) / 2

    return plus, minus


def IncorporateTexture(img, K):
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

    # Adquirindo Cb/Cr-mais e Cb/Cr-menos
    CbPlus, CbMinus = plus_minus(ReducedCb)
    CrPlus, CrMinus = plus_minus(ReducedCr)

    # Resizing Cb- to 1/4 of original size
    ReducedCbMinus = cv2.resize(
        CbMinus, (Sd1.shape[1], Sd1.shape[0]), interpolation=cv2.INTER_AREA)

    # Inverse Wavelet Transformation
    NewYSecondTry = (pywt.waverec2(
        (Sl, (Sh1, Sv1, ReducedCbMinus), (CrPlus, CbPlus, CrMinus)), 'db1'))

    # Result with simulation
    NewYSecondTry = sm.SimulateRealWorld(NewYSecondTry, K)
    return NewYSecondTry
