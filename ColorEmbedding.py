import cv2
import pywt
import numpy as np
import ColorSpaceTools as cst

#############################################################################################
#This module cointains the algorithm of the color codification which inserts the            #
#chrominance channels within the luminance channel by using the wavelet transform, creating #
#a texturized grayscale image.                                                              #
#############################################################################################


# Function that incorporate the texture in a RGB imagem
def incorporateTexture(image):
    # Transforming image to double type
    image = np.float32(image)

    # Changing domain to YCrCb
    image = cst.BGR2YCrCb(image)
    Y, Cr, Cb = cv2.split(image)

    # Wavelet Transformation in 2 levels
    (Sl, (Sh1, Sv1, Sd1), (Sh2, Sv2, Sd2)) = pywt.wavedec2(Y, 'db1', level=2)

    # Resizing layers
    reducedCb = cv2.resize(
        Cb, (Sd2.shape[1], Sd2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    reducedCr = cv2.resize(
        Cr, (Sv2.shape[1], Sv2.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Acquiring Cb/Cr-plus e Cb/Cr-minus
    CbPlus, CbMinus = cst.dividePlusMinus(reducedCb)
    CrPlus, CrMinus = cst.dividePlusMinus(reducedCr)

    # Resizing Cb- to 1/4 of original size
    reducedCbMinus = cv2.resize(
        CbMinus, (Sd1.shape[1], Sd1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Inverse Wavelet Transformation
    newYSecondTry = (pywt.waverec2(
        (Sl, (Sh1, Sv1, reducedCbMinus), (CrPlus, CbPlus, CrMinus)), 'db1'))

    return newYSecondTry
