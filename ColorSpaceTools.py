import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################################
#This module defines the functions that are useful in the manipulation of      #
#the color spaces of colored images                                            #
################################################################################


# Function that separates an array into a positive and a negative array
def dividePlusMinus(image):

    # Separate the image in positive(plus) and negative(minus)
    height, width = image.shape

    plus = np.zeros((height, width), np.float32)
    minus = np.zeros((height, width), np.float32)

    # Separate the image in positive(plus) and negative(minus)
    minus = (image - np.abs(image)) / 2
    plus = (image + np.abs(image)) / 2

    return plus, minus


# Adjust image saturation
def saturation(image, fit):
    image = np.float32(image)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)

    S *= fit
    S -= 255
    SPlus, SMinus = dividePlusMinus(S)
    S = SMinus + 255

    HSV = cv2.merge([H, S, V])
    image = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    return image


# Transform an RGB image to YCrCb image
def BGR2YCrCb(image):
    Y = image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299
    Cr = 0.713 * image[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * image[:, :, 0] - 0.564 * Y[:, :]

    return cv2.merge([Y, Cr, Cb])


# Transform an YCbCr image to BGR image
def YCrCb2BGR(image):
    R = ((image[:, :, 0]) + (1.402 * image[:, :, 1]))
    G = ((image[:, :, 0]) - (0.344 * image[:, :, 2]) - (0.714 * image[:, :, 1]))
    B = ((image[:, :, 0]) + (1.772 * image[:, :, 2]))

    return cv2.merge([B, G, R])


# Calculate PSNR's values
def getPSNR(true, pred):
    giantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
    giantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)

    return cv2.PSNR(giantTrueMatrix, giantPredMatrix)
