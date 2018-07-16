import cv2
import pywt
import numpy as np
import Transformations as tr


def RecoverColor(Image, pure, K):

    if pure == 1:
        # Removing haftone and going back to the original resolution (simulation case)
        Filtered = cv2.blur(Image, (K, K))
        Image = cv2.resize(Filtered, (int((1 / K) * Filtered.shape[0]), int(
            (1 / K) * Filtered.shape[1])), interpolation=cv2.INTER_AREA)

    # Wavelet Transformation in 2 levels
    (Sl, (Sh1, Sv1, Sd1), (Sh2, Sv2, Sd2)) = pywt.wavedec2(Image, 'db1', level=2)

    # Interpolating Sd1
    InterpolateSd1 = (cv2.resize(Sd1, dsize=(
        Sv2.shape[1], Sv2.shape[0]), interpolation=cv2.INTER_AREA))

    # Finding new Cb and Cr layers
    Cb = (np.abs(Sv2) - np.abs(InterpolateSd1))
    Cr = (np.abs(Sh2) - np.abs(Sd2))

    # Fulling some wavelet's layers with 0
    ZerosSd1 = np.zeros((Sd1.shape[0], Sd1.shape[1]))
    ZerosSh2 = np.zeros((Sh2.shape[0], Sh2.shape[1]))
    ZerosSv2 = np.zeros((Sv2.shape[0], Sv2.shape[1]))
    ZerosSd2 = np.zeros((Sd2.shape[0], Sd2.shape[1]))

    # Inverse wavelet transformation with zeros to get Y layer back
    FinalYSecondTry = (pywt.waverec2(
        (Sl, (Sh1, Sv1, ZerosSd1), (ZerosSh2, ZerosSv2, ZerosSd2)), 'db1'))

    # Resizing layers
    InterpolateCb = (cv2.resize(
        Cb, (2 * Cb.shape[1], 2 * Cb.shape[0]), interpolation=cv2.INTER_AREA))
    InterpolateCr = (cv2.resize(
        Cr, (2 * Cr.shape[1], 2 * Cr.shape[0]), interpolation=cv2.INTER_AREA))

    # Building final image
    finalimage = cv2.merge(
        ((FinalYSecondTry), (InterpolateCr), (InterpolateCb)))

    # Returning to RGB domain
    finalimage = tr.YCbCr2BGR((finalimage))

    return finalimage
