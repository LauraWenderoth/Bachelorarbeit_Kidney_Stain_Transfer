from PIL import Image
import numpy as np

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
