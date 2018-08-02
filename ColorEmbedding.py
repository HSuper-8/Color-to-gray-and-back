import cv2
import pywt
import numpy as np
import Simulation as sm
import Transformations as tr

# Function that separates an array into a positive and a negative array  
def DividePlusMinus(img):

    # Separate the image in positive(plus) and negative(minus)
    height, width = img.shape

    plus = np.zeros((height, width), np.float32)
    minus = np.zeros((height, width), np.float32)
    
    # Separate the image in positive(plus) and negative(minus)

    minus = (img - np.abs(img)) / 2
    plus = (img + np.abs(img)) / 2

    return plus, minus


# Function that incorporate the texture in a RGB imagem
def IncorporateTexture(img):
    # Transforming image to double type
    img = np.float32(img)

    # Changing domain to YCrCb
    img = tr.BGR2YCrCb(img)
    Y, Cr, Cb = cv2.split(img)

    # Wavelet Transformation in 2 levels
    (Sl, (Sh1, Sv1, Sd1), (Sh2, Sv2, Sd2)) = pywt.wavedec2(Y, 'db1', level=2)

    # Resizing layers
    ReducedCb = cv2.resize(
        Cb, (Sd2.shape[1], Sd2.shape[0]), interpolation=cv2.INTER_AREA)
    ReducedCr = cv2.resize(
        Cr, (Sv2.shape[1], Sv2.shape[0]), interpolation=cv2.INTER_AREA)

    # Acquiring Cb/Cr-plus e Cb/Cr-minus
    CbPlus, CbMinus = DividePlusMinus(ReducedCb)
    CrPlus, CrMinus = DividePlusMinus(ReducedCr)

    # Resizing Cb- to 1/4 of original size
    ReducedCbMinus = cv2.resize(
        CbMinus, (Sd1.shape[1], Sd1.shape[0]), interpolation=cv2.INTER_AREA)

    # Inverse Wavelet Transformation
    NewYSecondTry = (pywt.waverec2(
        (Sl, (Sh1, Sv1, ReducedCbMinus), (CrPlus, CbPlus, CrMinus)), 'db1'))
    
    return NewYSecondTry
