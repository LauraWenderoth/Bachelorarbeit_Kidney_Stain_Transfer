from PIL import Image
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

#Laura
def pad_image_to_size(img, patch_size):
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

def remove_pad_from_image(img, orignial_img,patch_size):
    if orignial_img.shape[0] or orignial_img.shape[1] < patch_size:
        difference_x = patch_size - orignial_img.shape[0]
        pad_x1 = difference_x // 2
        pad_x2 = difference_x // 2
        if not difference_x % 2 == 0:
            pad_x2 += 1

        difference_y = patch_size - orignial_img.shape[1]
        pad_y1 = difference_y // 2
        pad_y2 = difference_y // 2
        if not difference_y % 2 == 0:
            pad_y2 += 1

        i = img[pad_x1:-pad_x2,pad_y1:-pad_y2]
        return i
    else:
        print("WARNING! Not implemented in laplacian_scaling")



#https://theailearner.com/tag/laplacian-pyramid-opencv/
def calculate_gaussian_pyramids(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr

def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


# Then calculate the Laplacian pyramid
def calculate_laplacian_pyramids(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr

if __name__ == '__main__':
    # path = "/home/laurawenderoth/Documents/kidney_microscopy/data/PAS/CKD154-003-PAS-fully-aligned.png"
    # img = Image.open(path)
    # img = np.array(img)
    # #new image
    # path = "/home/laurawenderoth/Documents/kidney_microscopy/data/IF/CKD154-003-IF-fully-aligned.png"
    # IF = Image.open(path)
    # IF = np.array(IF)
    # lower = img.copy()
    # # Create a Gaussian Pyramid
    # gaussian_pyr = [lower]
    # for i in range(5):
    #     lower = cv2.pyrDown(lower)
    #     gaussian_pyr.append(lower)
    # # Last level of Gaussian remains same in Laplacian
    # laplacian_top = gaussian_pyr[-1]
    #
    # # Create a Laplacian Pyramid
    # laplacian_pyr = [laplacian_top]
    # for i in range(5, 0, -1):
    #     size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
    #     gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
    #     laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
    #     laplacian_pyr.append(laplacian)
    # for g in laplacian_pyr:
    #     g = np.array(g, dtype=np.uint8)
    #     g= g/g.max()
    #     print(g.min(),g.max())
    #     plt.imshow(g)
    #     plt.show()

    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/PAS/CKD154-003-PAS-fully-aligned.png"
    img = Image.open(path)
    img = np.array(img)
    # down_img = downsampling(img)
    # up_img = laplacian_upsampling(img,down_img)

    expo = np.array(img).shape[0].bit_length()
    num_levels = expo - 8
    print(img.shape)
    img_pad = pad_image_to_size(img,2048)
    lower = img_pad.copy()
    gaussian_pyramids = calculate_gaussian_pyramids(np.array(lower), num_levels)
    g = gaussian_pyramids[-1]
    g = np.array(g, dtype=np.uint8)
    print(g.shape)
    #new image
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/IF/CKD154-003-IF-fully-aligned.png"
    IF = Image.open(path)
    IF = np.array(IF)
    img_pad_if = pad_image_to_size(IF, 2048)
    lower_if = img_pad_if.copy()
    gaussian_pyramids_if = calculate_gaussian_pyramids(np.array(lower_if), num_levels)
    g_if = gaussian_pyramids_if[-1]
    g_if= np.array(g_if, dtype="float32")

    laplacian_pyramid = calculate_laplacian_pyramids(gaussian_pyramids)
    l_if_pyramids = laplacian_pyramid.copy()
    l_if_pyramids[0] = g_if
    laplacian_lst = reconstruct(l_if_pyramids)
    fertiges_Bild = laplacian_lst[-1]
    #veränderung dtype von float32 zu np.uint8
    fertiges_Bild = np.array(fertiges_Bild, dtype=np.uint8)
    img_without_pad = remove_pad_from_image(fertiges_Bild,img,2048)
    plt.imshow(img)
    plt.title("orignial image PAS")
    plt.show()
    plt.imshow(img_pad)
    plt.title("padded image PAS")
    plt.show()
    g_if = np.array(g_if, dtype=np.uint8)
    plt.imshow(g_if)
    plt.title("downsampled image IF")
    plt.show()
    plt.imshow(fertiges_Bild)
    plt.title("upsampled image IF with PAS Laplacian pyramids")
    plt.show()
    plt.imshow(img_without_pad)
    plt.title("upsampled without padding")
    plt.show()

'''
for g in gaussian_pyramid:
    g = np.array(g, dtype=np.uint8)
    plt.imshow(g)
    plt.show()
    
fig = plt.figure(figsize=(10,5))
    fig.add_subplot(1, 4, 1)
    plt.imshow(img)
    fig.add_subplot(1, 4, 2)
    plt.imshow(g)
    fig.add_subplot(1, 4, 3)
    plt.imshow(fertiges_Bild)
    fig.add_subplot(1, 4, 4)
    plt.imshow(img_without_pad)
    plt.show()
'''