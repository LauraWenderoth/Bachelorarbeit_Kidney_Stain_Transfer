from PIL import Image
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

#Laura
def pad_image_to_size(img, patch_size):
    img = np.array(img)
    if img.shape[0] or img.shape[1] < patch_size:
        difference_x = patch_size - img.shape[0]
        pad_x1 = difference_x // 2
        pad_x2 = difference_x // 2
        if not difference_x % 2 == 0:
            pad_x2 += 1

        difference_y = patch_size - img.shape[1]
        pad_y1 = difference_y // 2
        pad_y2 = difference_y // 2
        if not difference_y % 2 == 0:
            pad_y2 += 1

        i = np.pad(img, ((pad_x1, pad_x2), (pad_y1, pad_y2), (0, 0)), 'symmetric')
        return Image.fromarray(i)
    else:
        print("WARNING! Not implemented in laplacian_scaling")

if __name__ == '__main__':
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/PAS/CKD154-003-PAS-fully-aligned.png"
    img = Image.open(path)
    img = np.array(img)

    #new image
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/IF/CKD154-003-IF-fully-aligned.png"
    IF = Image.open(path)
    IF = np.array(IF)

    expo = np.array(img).shape[0].bit_length()
    patchsize = 2**expo
    number_patches = patchsize//256

    img_pad = pad_image_to_size(img,patchsize)
    patches = []
    for x in range(number_patches):
        for y in range(number_patches):
            patch = img_pad[x*256:x*256+256,y*256:y*256+256]
            patches.append(patch)

    #plot
    fig = plt.figure(figsize=(20, 20))
    columns = number_patches
    rows = number_patches
    for i in range(1, columns * rows + 1):
        img = patches[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()





