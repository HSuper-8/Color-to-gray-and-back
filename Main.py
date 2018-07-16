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

    # Textured Images without simulations of the real world
    pathlib.Path('./PureImagesTextures').mkdir(parents=True, exist_ok=True)

    # Textured Images with simulations of the real world
    pathlib.Path(
        './ImagesTexturesSimulations').mkdir(parents=True, exist_ok=True)

    # Final Images after the process without simulations of the real world
    pathlib.Path('./ImagesResultsPure').mkdir(parents=True, exist_ok=True)

    # Final Images after the process with simulations of the real world
    pathlib.Path('./ImagesResultsReal').mkdir(parents=True, exist_ok=True)

    # Reading original images
    imagesToTrain = [cv2.imread(file, 3)
                     for file in np.sort(glob.glob("./Images/*.png"))]

    # Incoporating colors in the gray version of the image, adding texture
    for i in range(0, 7):
        Image = ce.IncorporateTexture(imagesToTrain[i], 0, 4)
        # Without simulation version
        cv2.imwrite("./PureImagesTextures/%d.png" % i, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # With simulation version
        Image = ce.IncorporateTexture(imagesToTrain[i], 1, 4)
        cv2.imwrite("./ImagesTexturesSimulations/%d.png" % i, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Reading textured images(both type)
    PureImages = [cv2.imread(file, 0) for file in np.sort(
        glob.glob("./PureImagesTextures/*.png"))]
    RealImages = [cv2.imread(file, 0) for file in np.sort(
        glob.glob("./ImagesTexturesSimulations/*.png"))]

    # Trying to re-create original images
    for i in range(0, 7):
        Image = cr.RecoverColor(PureImages[i], 0, 4)
        # Without simulation version
        cv2.imwrite("./ImagesResultsPure/%d.png" % i, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        Image = cr.RecoverColor(RealImages[i], 1, 4)
        # With simulation version
        cv2.imwrite("./ImagesResultsReal/%d.png" % i, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Reading restored images with simulations of the real world
    ResultsReal = [cv2.imread(file, 3) for file in np.sort(
        glob.glob("./ImagesResultsReal/*.png"))]

    # Calculating PSNR's values of the real results
    for i in range(0, 7):
        print('Image %d PSNR: %lf' %
              (i, tr.getPSNR(imagesToTrain[i], ResultsReal[i])))


if __name__ == '__main__':
    main()
