from PIL import Image
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/PAS/CKD154-003-PAS-fully-aligned.png"
    img = Image.open(path)
    img = np.array(img)

    #new image
    path = "/home/laurawenderoth/Documents/kidney_microscopy/data/IF/CKD154-003-IF-fully-aligned.png"
    IF = Image.open(path)
    IF = np.array(IF)

