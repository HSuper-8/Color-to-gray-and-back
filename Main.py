import cv2
import glob
import pathlib
import numpy as np
import ColorEmbedding as ce
import ColorRecovery as cr
import Simulation as sm
import Transformations as tr
from PIL import Image, ImageEnhance

# This is the main program of the Color to Gray and Back algorithm

def main():
    # Creating directories to save images

    # Textured Images with simulations of the real world
    pathlib.Path(
        './ImagesTextures').mkdir(parents=True, exist_ok=True)

    # Final Images after the process with simulations of the real world
    pathlib.Path('./ImagesResults').mkdir(parents=True, exist_ok=True)

    k = int(input("Enter a resize order\n"))

    i = 0
    for file in np.sort(glob.glob("./Images/*.png")):
        Image = cv2.imread(file, 3)

        Image = ce.IncorporateTexture(Image, k)

        Image = sm.SimulateRealWorld(Image, k)
        cv2.imwrite("./ImagesTextures/%d.png" % i, Image)

        # Trying to re-create original images
        Image = cr.RecoverColor(Image, k)
        cv2.imwrite("./ImagesResults/%d.png" % i, Image)
        i += 1

    # Reading original images
    Originals = [cv2.imread(file, 3) for file in np.sort(
        glob.glob("./Images/*.png"))]


    cr.Saturation(i)

    # Reading restored images with simulations of the real world
    Results = [cv2.imread(file, 3) for file in np.sort(
        glob.glob("./ImagesResults/*.png"))]    
    
    for k in range (0, i):
        # Calculating PSNR's values of the real results
        print('Image %d PSNR: %lf' %
              (k, tr.getPSNR(Originals[k], Results[k])))


if __name__ == '__main__':
    main()
