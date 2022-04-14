### ANN KATRIN°°!

import cv2
from PIL import Image
import numpy as np

def downsampling(original):
    """
    Function for downsampling multiple images to 256x256 pixels.
    If an image is smaller than 256x256 the original is returned
    :param originals: Images to downscale using a Gaussian pyramid
    :return: Downsampled images
    """

    # assert original.shape[0] == original.shape[1], "Images have to be squares"
    original = np.array(original)
    if original.shape[0] != original.shape[1]:
        size = min(original.shape[0], original.shape[1])
        original = original[:size][:size][:]
    # Find largest power of two that is less than the image size
    expo = original.shape[0].bit_length() - 1
    # Make sure image isn't smaller than 256x256 pixels
    if expo < 8:
        return original
    img = cv2.resize(original, dsize=(2 ** expo, 2 ** expo), interpolation=cv2.INTER_CUBIC)
    g = img.copy()
    # Resize image to 256x256 (=2**8)
    for i in range(expo - 8):
        g = cv2.pyrDown(g)
    downsampled_original = Image.fromarray(np.array(g))
    return downsampled_original



def laplacian_upsampling(original, input):
    """
    Perform upsampling of generated images as explained by Engin in the CycleDehaze paper (2018)
    :param originals: Input the images were generated from (original size)
    :param inputs: Generated images (small size)
    :param original_shape: Shape of the original image
    :return: Generated images (original size
    """

    # assert original.shape[0] == original.shape[1], "Images have to be squares"
    original = np.array(original)
    input = np.array(input)
    size = min(original.shape[0], original.shape[1])
    if original.shape[0] != original.shape[1]:
        original = original[:size][:size][:]
    # Find largest power of two that is less than the image size
    expo = original.shape[0].bit_length() - 1
    img = cv2.resize(original, dsize=(2 ** expo, 2 ** expo), interpolation=cv2.INTER_CUBIC)

    # Calculate laplacian pyramid
    # Downsample original
    g_pyramid = []
    ga = img.copy()
    g_pyramid.append(ga.copy())
    # Downsample image to 256x256 (=2**8)
    for i in range(expo - 8):
        ga = cv2.pyrDown(ga)
        g_pyramid.append(ga.copy())
    l_pyramid = []
    for i in range(expo - 8):
        lap = cv2.subtract(g_pyramid[i], cv2.pyrUp(g_pyramid[i + 1]))
        l_pyramid.append(lap.copy())
    # Last element of g pyramid is last element of l pyramid
    l_pyramid.append(g_pyramid[-1].copy())
    # Laplacian upsampling based on laplacian pyramid of the original
    up_pyramid = []
    up = input
    up_pyramid.append(up.copy())
    for i in range(expo - 8):
        up = cv2.pyrUp(up) + l_pyramid[expo - (9 + i)]
        up_pyramid.append(up.copy())
    upsampled = up_pyramid[-1].copy()
    upsampled = np.clip(upsampled, -1, 1)
    # Re-size image to have original size
    upsampled = cv2.resize(upsampled, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    # Have to cut off padding:
    # upsampled_cut = remove_minimal_pad(upsampled, (size, size))
    upsampled_input = Image.fromarray(upsampled)
    return upsampled_input
