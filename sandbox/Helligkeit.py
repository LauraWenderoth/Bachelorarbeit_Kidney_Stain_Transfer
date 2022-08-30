import glob
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hsv

def brightness(path):
    rgb_img = Image.open(path)
    rgb_img = np.asarray(rgb_img)
    hsv_img = rgb2hsv(rgb_img)
    value_img = hsv_img[:, :, 2]
    plt.imshow(rgb_img)
    plt.show()
    plt.imshow(value_img,cmap="gray")
    plt.show()
    mean = value_img.mean()
    print(mean)
    return mean

bad_paths = "/home/laurawenderoth/Documents/Bachelorarbeit/fails/test_pix2pix_PAS2IF_L1reset_patches_seed_957_epoch105/real_A/bad/*.png"
paths = "/home/laurawenderoth/Documents/Bachelorarbeit/fails/test_pix2pix_PAS2IF_L1reset_patches_seed_957_epoch105/real_A/*.png"
image_path = glob.glob(paths,recursive=True)
bad_image_path = glob.glob(bad_paths,recursive=True)

bad_images_mean = []
normal_images_mean = []
for path in bad_image_path:
    mean = brightness(path)
    bad_images_mean.append(mean)

print("####################################")
for path in image_path:
    mean = brightness(path)
    normal_images_mean.append(mean)
bad_images_mean = np.asarray(bad_images_mean)
print("mean of bad images", bad_images_mean.mean(), "std", bad_images_mean.std())
normal_images_mean = np.asarray(normal_images_mean)
print("mean of bad images", normal_images_mean.mean(),"std", normal_images_mean.std())