import cv2
import numpy as np

################################################################################
#This module defines the functions that are useful in the manipulation of      #
#the color spaces of colored images                                            #
################################################################################


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


# Increases image saturation
def Saturation(img):
    img = np.float32(img)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)

    S *= 1.2
    S -= 255
    SPlus, SMinus = DividePlusMinus(S)
    S = SMinus + 255

    HSV = cv2.merge([H, S, V])
    img = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    return img


# Transform an RGB image to YCrCb image
def BGR2YCrCb(img):
    Y = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
    Cr = 0.713 * img[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * img[:, :, 0] - 0.564 * Y[:, :]

    return cv2.merge([Y, Cr, Cb])


# Transform an YCbCr image to BGR image
def YCrCb2BGR(img):
    R = ((img[:, :, 0]) + (1.402 * img[:, :, 1]))
    G = ((img[:, :, 0]) - (0.344 * img[:, :, 2]) - (0.714 * img[:, :, 1]))
    B = ((img[:, :, 0]) + (1.772 * img[:, :, 2]))

    return cv2.merge([B, G, R])


# Calculate PSNR's values
def getPSNR(true, pred):
    GiantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
    GiantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)

    return cv2.PSNR(GiantTrueMatrix, GiantPredMatrix)
