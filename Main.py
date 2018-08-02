import cv2
import glob
import pathlib
import numpy as np
import ColorEmbedding as ce
import ColorRecovery as cr
import Simulation as sm
import Transformations as tr

# This is the main program of the Color to Gray and Back algorithm
def main():
    # Creating directories to save images

    # Textured Images with simulations of the real world
    pathlib.Path(
        './ImagesTextures').mkdir(parents=True, exist_ok=True)

    # Final Images after the process with simulations of the real world
    pathlib.Path('./ImagesResults').mkdir(parents=True, exist_ok=True)

    simulation = bool(int(
        input("\nEnter (1) or (0) for the option:\n(1) With Simulation\n(0) No Simulation\n")))
    if(simulation):
        k = int(input("Enter a resize order\n"))    

    for file in np.sort(glob.glob("Images/*.png")):
        print("Imagem %s..." % file[7:],)
        Image = cv2.imread('Images/%s' % file[7:])

        Image = ce.IncorporateTexture(Image)
        if(simulation):
            print("Simulando distorção por impressão...")
            Image = sm.SimulateRealWorld(Image, k)
        cv2.imwrite("ImagesTextures/%s" % file[7:], Image)
    
        # Trying to re-create original images
        Image = cv2.imread('ImagesTextures/%s' % file[7:], 0)
        Image = cr.RecoverColor(Image)
        cv2.imwrite("ImagesResults/%s" % file[7:], Image)

        if(simulation):
            tr.Saturation()

        # Reading restored images with simulations of the real world
        Result = cv2.imread('ImagesResults/%s' % file[7:])

        # Calculating PSNR's values of the real results
        Original = cv2.imread("Images/%s" % file[7:])
        print('PSNR: %lf' % tr.getPSNR(Original, Result))
        
if __name__ == '__main__':
    main()
