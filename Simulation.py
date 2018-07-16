import cv2
import Transformations as tr


def SimulateRealWorld(image, K):

    # Scaling and Haftoning image simulating print and scaning in real life
    image = cv2.resize(
        image, (K * image.shape[0], K * image.shape[1]), interpolation=cv2.INTER_AREA)

    # Inserting error in the image
    image = tr.ErrorDiffusion(image)

    return image
