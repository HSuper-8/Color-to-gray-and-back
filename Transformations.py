import cv2
import numpy as np


# Transform an RGB image to YCbCr image
def RGB2YCBCR(img):
    Y = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
    Cr = 0.713 * img[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * img[:, :, 0] - 0.564 * Y[:, :]

    return cv2.merge([Y, Cr, Cb])


# Transform an YCbCr image to RGB image
def YCbCr2BGR(img):
    R = np.float32(((img[:, :, 0]) + (1.402 * img[:, :, 1])))
    G = np.float32(
        ((img[:, :, 0]) - (0.344 * img[:, :, 2]) - (0.714 * img[:, :, 1])))
    B = np.float32(((img[:, :, 0]) + (1.772 * img[:, :, 2])))
    return cv2.merge([B, G, R])


# Calculate PSNR's values
def getPSNR(true, pred):
    GiantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
    GiantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)

    return cv2.PSNR(GiantTrueMatrix, GiantPredMatrix)
