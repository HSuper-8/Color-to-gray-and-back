import cv2
import numpy as np
import glob


# increases image saturation
def Saturation(img):
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img)

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            S[x, y] = 1.2 * S[x, y]
            if S[x, y] > 255:
                S[x, y] = 255
    img[:, :, 1] = S
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


# Transform an RGB image to YCbCr image
def BGR2YCrCb(img):
    Y = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
    Cr = 0.713 * img[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * img[:, :, 0] - 0.564 * Y[:, :]

    return cv2.merge([Y, Cr, Cb])


# Transform an YCbCr image to RGB image
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
