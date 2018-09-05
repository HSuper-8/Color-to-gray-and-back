import cv2
import pywt
import numpy as np
import ColorSpaceTools as cst

###################################################################################
#This module contains the algorithm responsible for the color decodification      #
#of a textured image, recovering the original chrominance channels embbeded       #
#inside the image.                                                                #
###################################################################################


# Function that recovers the color of a gray imagem with embedded texture
def recoverColor(image):

    # Wavelet Transformation in 2 levels
    (Sl, (Sh1, Sv1, Sd1), (Sh2, Sv2, Sd2)) = pywt.wavedec2(image, 'db1', level=2)

    # Interpolating Sd1
    interpolateSd1 = (cv2.resize(Sd1, dsize=(
        Sv2.shape[1], Sv2.shape[0]), interpolation=cv2.INTER_LANCZOS4))

    # Finding new Cb and Cr layers
    Cb = (np.abs(Sv2) - np.abs(interpolateSd1))
    Cr = (np.abs(Sh2) - np.abs(Sd2))

    # Fulling some wavelet's layers with 0
    zerosSd1 = np.zeros((Sd1.shape[0], Sd1.shape[1]))
    zerosSh2 = np.zeros((Sh2.shape[0], Sh2.shape[1]))
    zerosSv2 = np.zeros((Sv2.shape[0], Sv2.shape[1]))
    zerosSd2 = np.zeros((Sd2.shape[0], Sd2.shape[1]))

    # Inverse wavelet transformation with zeros to get Y layer back
    finalY = (pywt.waverec2(
        (Sl, (Sh1, Sv1, zerosSd1), (zerosSh2, zerosSv2, zerosSd2)), 'db1'))

    # Resizing layers
    interpolateCb = (cv2.resize(
        Cb, (2 * Cb.shape[1], 2 * Cb.shape[0]), interpolation=cv2.INTER_LANCZOS4))
    interpolateCr = (cv2.resize(
        Cr, (2 * Cr.shape[1], 2 * Cr.shape[0]), interpolation=cv2.INTER_LANCZOS4))

    # Building final image
    finalImage = cv2.merge(
        ((finalY), (interpolateCr), (interpolateCb)))

    # Returning to RGB domain
    finalImage = cst.YCrCb2BGR((finalImage))

    return finalImage
