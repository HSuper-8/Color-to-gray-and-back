import cv2
import numpy as np

# increases image saturation
def Saturation(Result):
    Result = cv2.cvtColor(Result, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(Result)
    for x in range(0, Result.shape[0]):
        for y in range(0, Result.shape[1]):
            S[x,y] = 1.2 * S[x,y]
            if S[x, y] > 1:
                S[x, y] = 1.0
    Result = cv2.merge([H, S, V])
    Result = cv2.cvtColor(Result, cv2.COLOR_HSV2BGR)
    return Result

# Transform an RGB image to YCbCr image
def BGR2YCrCb(img):
    img = np.float32(img)
    Y = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
    Cr = 0.713 * img[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * img[:, :, 0] - 0.564 * Y[:, :]

    return cv2.merge([Y, Cr, Cb])


# Transform an YCbCr image to RGB image
def YCrCb2BGR(img):
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
