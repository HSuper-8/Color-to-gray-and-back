import cv2
import glob
import pathlib
import numpy as np
import ColorEmbedding as ce
import ColorRecovery as cr
import Transformations as tr

# This is the main program of the second attempt


def main():

    # Creating directories to save images

    # Textured Images with simulations of the real world
    pathlib.Path(
        './ImagesTextures').mkdir(parents=True, exist_ok=True)

    # Final Images after the process with simulations of the real world
    pathlib.Path('./ImagesResults').mkdir(parents=True, exist_ok=True)

    k = int(input("Enter a resize order\n"))

    for file in np.sort(glob.glob("./Images/*.png")):
        Image = cv2.imread(file, 3)

        Image = ce.IncorporateTexture(Image, 1, k)
        cv2.imwrite("./ImagesTextures/%d.png" % i, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Trying to re-create original images
        Image = cr.RecoverColor(PureImages[i], 1, k)
        cv2.imwrite("./ImagesResults/%d.png" % i, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Reading restored images with simulations of the real world
    Results = [cv2.imread(file, 3) for file in np.sort(
        glob.glob("./ImagesResults/*.png"))]
    
    # Reading restored images with simulations of the real world
    Originals = [cv2.imread(file, 3) for file in np.sort(
        glob.glob("./Images/*.png"))]    

    # Calculating PSNR's values of the real results
    for i in range(0, 7):
        print('Image %d PSNR: %lf' %
              (i, tr.getPSNR(Originals[i], Results[i])))


if __name__ == '__main__':
    main()
