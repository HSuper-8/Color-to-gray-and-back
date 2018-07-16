import cv2
import Transformations as tr


# Função que aplica uma difusão de erros usando o método de Floyd-Steinberg.
def ErrorDiffusion(img, size=(1, 1)):
        img = np.float32(img)
        #coloca uma borda preta na imagem
        img_exp = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
        img_exp[1:img.shape[0] + 1, 1:img.shape[1] + 1] = img[:, :]

        for y in range(1, size[1] + 1):
                for x in range(1, size[0] + 1):
                        old_pixel = img_exp[x, y]
                        if old_pixel > 127:
                            img_exp[x, y] = 255
                        else:
                            img_exp[x, y] = 0
                        quant_error = old_pixel - img_exp[x, y]
                        img_exp[x + 1, y] = img_exp[x + 1, y] + 7 / 16.0 * quant_error
                        img_exp[x - 1, y + 1] = img_exp[x - 1, y + 1] + 3 / 16.0 * quant_error
                        img_exp[x, y + 1] = img_exp[x, y + 1] + 5 / 16.0 * quant_error
                        img_exp[x + 1, y + 1] = img_exp[x + 1, y + 1] + 1 / 16.0 * quant_error

        #retira borda
        img[:, :] = img_exp[1:img.shape[0] + 1, 1:img.shape[1] + 1]

        return np.uint8(img)


def SimulateRealWorld(image, K):

    # Scaling and Haftoning image simulating print and scaning in real life
    image = cv2.resize(
        image, (K * image.shape[0], K * image.shape[1]), interpolation=cv2.INTER_AREA)

    # Inserting error in the image
    image = tr.ErrorDiffusion(image)

    return image
