import cv2
import glob
import pathlib
import numpy as np
import ColorEmbedding as ce
import ColorRecovery as cr
import Simulation as sm
import Transformations as tr
import  PIL
from PIL import Image

# This is the main program of the second attempt


def main():
    simulation = 1
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

        if(simulation == 1):
            Image = sm.SimulateRealWorld(Image, k)
        cv2.imwrite("./ImagesTextures/%d.png" % i, Image)

        # Trying to re-create original images
        Image = cr.RecoverColor(Image, k)
        cv2.imwrite("./ImagesResults/%d.png" % i, Image)
        i += 1

    # Reading restored images with simulations of the real world
    Results = [cv2.imread(file, 3) for file in np.sort(
        glob.glob("./ImagesResults/*.png"))]

    #Reading restored images with simulations of the real world
    #i = 0
    #Results = np.array([])
    #for file in np.sort(glob.glob("./ImagesResults/*.png")):
        #Results.append = PIL.Image.open(file)
        #convert = PIL.ImageEnhance.Color(Results[i])
        #Results[i] = convert.enhance(0.5) 
        #Results[i].save("./ImagesResults/%d.png" % i,)      
        #i += 1
    
    # Calculating PSNR's values of the real results
    i = 0
    for file in np.sort(glob.glob("./Images/*.png")):
        Original = cv2.imread(file, 3)
        print('Image %d PSNR: %lf' %
              (i, tr.getPSNR(Original, Results[i])))
        i += 1


if __name__ == '__main__':
    main()
