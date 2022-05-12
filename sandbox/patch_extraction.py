from PIL import Image
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

def downsampling(img, patch_size):
    expo = np.array(img).shape[0].bit_length()
    num_levels = expo - patch_size.bit_length()
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    g = gaussian_pyr[-1]
    return g.astype(np.uint8)

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
        return i
    else:
        print("WARNING! Not implemented in laplacian_scaling")

if __name__ == '__main__':
    patches_per_width = 2
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/PAS/CKD154-003-PAS-fully-aligned.png"
    img = Image.open(path)
    img = np.array(img)

    #new image
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/IF/CKD154-003-IF-fully-aligned.png"
    IF = Image.open(path)
    IF = np.array(IF)

    expo = np.array(img).shape[0].bit_length()
    patchsize = 2**expo

    img_pad = pad_image_to_size(img,patchsize)
    patch_size = int(2048 / patches_per_width)
    patches = []
    for x in range(patches_per_width):
        for y in range(patches_per_width):
            patch = img_pad[x * patch_size:x * patch_size + patch_size, y * patch_size:y * patch_size + patch_size]
            if patch_size != 256:
                patch = downsampling(np.array(patch), 256)
            patches.append(patch)

    #plot
    fig = plt.figure(figsize=(20, 20))
    columns = patches_per_width
    rows = patches_per_width
    for i in range(1, columns * rows + 1):
        img = patches[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()





