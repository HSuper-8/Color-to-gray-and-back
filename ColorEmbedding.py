import cv2
import pywt
import numpy as np
import ColorSpaceTools as cst

#############################################################################################
#Esse modulo contém o algoritmo de codificação das cores incorporando os canais de          # 
#crominancia no canal de luminancia através da transformada wavelet, gerando                #
#uma imagem texturizada em níveis de cinza.                                                 #
#                                                                                           #
#############################################################################################


# Function that incorporate the texture in a RGB imagem
def IncorporateTexture(img):
    # Transforming image to double type
    img = np.float32(img)

    # Changing domain to YCrCb
    img = cst.BGR2YCrCb(img)
    Y, Cr, Cb = cv2.split(img)

    # Wavelet Transformation in 2 levels
    (Sl, (Sh1, Sv1, Sd1), (Sh2, Sv2, Sd2)) = pywt.wavedec2(Y, 'db1', level=2)

    # Resizing layers
    ReducedCb = cv2.resize(
        Cb, (Sd2.shape[1], Sd2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    ReducedCr = cv2.resize(
        Cr, (Sv2.shape[1], Sv2.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Acquiring Cb/Cr-plus e Cb/Cr-minus
    CbPlus, CbMinus = cst.DividePlusMinus(ReducedCb)
    CrPlus, CrMinus = cst.DividePlusMinus(ReducedCr)

    # Resizing Cb- to 1/4 of original size
    ReducedCbMinus = cv2.resize(
        CbMinus, (Sd1.shape[1], Sd1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Inverse Wavelet Transformation
    NewYSecondTry = (pywt.waverec2(
        (Sl, (Sh1, Sv1, ReducedCbMinus), (CrPlus, CbPlus, CrMinus)), 'db1'))

    return NewYSecondTry
