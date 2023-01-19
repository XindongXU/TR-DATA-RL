import cv2
import matplotlib.pyplot as plt
import numpy as np


# the image must be in the bgr format
def find_stick_HSV(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_H = image[:, :, 0]
    image_S = image[:, :, 1]
    image_V = image[:, :, 2]

    # plt.imshow(image)
    # plt.show()

    # plt.imshow(image_H, cmap='gray')
    # plt.show()

    # plt.imshow(image_S, cmap='gray')
    # plt.show()

    # plt.imshow(image_V, cmap='gray')
    # plt.show()

    mask = np.logical_and(np.logical_and(image_H > 150, image_H < 180), image_S > 50, np.logical_and(image_V < 50, image_V > 30))
    # mask = np.logical_and(np.logical_and(image_H > 0, image_H < 120), image_S > 85, np.logical_and(image_V < 123, image_V > 0))

    # image[mask] = []
    # plt.imshow(image)
    # plt.show()

    return mask


# the image must be in the bgr format
def find_stick_BGR(image):

    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]

    # plt.imshow(image)
    # plt.show()

    # plt.imshow(image_R, cmap='gray')
    # plt.show()

    # plt.imshow(image_G, cmap='gray')
    # plt.show()

    # plt.imshow(image_B, cmap='gray')
    # plt.show()

    mask = np.logical_and(image_R < 45, image_G > 60, image_B > 50)

    # plt.imshow(mask, cmap = 'gray')
    # plt.show()

    # image[mask, :] = [255, 0, 255]
    # plt.imshow(image)
    # plt.show()

    return mask

# the image must be in the rgb format
def find_stick_LAB(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    image_L = image[:, :, 0]
    image_A = image[:, :, 1]
    image_B = image[:, :, 2]

    # plt.imshow(image)
    # plt.show()

    # plt.imshow(image_L, cmap='gray')
    # plt.show()

    # plt.imshow(image_A, cmap='gray')
    # plt.show()

    # plt.imshow(image_B, cmap='gray')
    # plt.show()

    mask = image_A > 220

    # plt.imshow(mask, cmap = 'gray')
    # plt.show()

    # image[mask, :] = [255, 0, 255]
    # plt.imshow(image)
    # plt.show()

    return mask


if __name__ == "__main__":
    print("hei")
