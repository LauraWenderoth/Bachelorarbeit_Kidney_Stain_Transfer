import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def calculate_gaussian_pyramids(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


# Laura

if __name__ == "__main__":
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/PAS/CKD154-003-PAS-fully-aligned.png"
    img = Image.open(path)
    img = np.array(img)
    width, heigth = img.shape[:2]

    # new image
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/IF/CKD154-003-IF-fully-aligned.png"
    IF = Image.open(path)
    IF = np.array(IF)
    patches = []
    names = []
    gaussian_pyramids = calculate_gaussian_pyramids(np.array(IF), 3)
    g = gaussian_pyramids[-1]
    g = np.uint8(g)
    patches.append(img)
    names.append("normal PAS")
    patches.append(IF)
    names.append("normal IF")
    patches.append(g)
    names.append("down IF")

    up = Image.fromarray(g)
    up = up.resize((heigth, width), resample=Image.BICUBIC)
    up = np.array(up)

    patches.append(up)
    names.append("up IF")

    Differenz = IF - up
    patches.append(Differenz)
    names.append("IF-UP")

    # plot
    columns = len(patches)
    rows = 1
    fig = plt.figure(figsize=(columns * 3, rows * 5))
    for i in range(1, columns * rows + 1):
        img = patches[i - 1]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(names[i - 1])
        plt.imshow(img)
    plt.show()
