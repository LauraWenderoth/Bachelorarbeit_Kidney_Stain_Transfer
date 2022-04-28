import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from data.patch_extraction import pad_image_to_size
import cv2
from options.train_options import TrainOptions
from data import create_dataset

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        patches_per_width = 4
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        path_index = index // patches_per_width**2
        A_path = self.A_paths[path_index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = path_index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        #slice part
        img_pad_A = pad_image_to_size(A_img, 2048)
        img_pad_B = pad_image_to_size(B_img, 2048)

        patch_index = index % patches_per_width**2
        x = patch_index//patches_per_width
        y = patch_index % patches_per_width
        patch_size = int(2048/patches_per_width)
        A = img_pad_A[x * patch_size:x * patch_size + patch_size, y * patch_size:y * patch_size + patch_size]
        B = img_pad_B[x * patch_size:x * patch_size + patch_size, y * patch_size:y * patch_size + patch_size]
        if patch_size != 256:
            A = downsampling(np.array(A), 256)
            B = downsampling(np.array(B), 256)
        #convert to PIL
        A = Image.fromarray(A)
        B = Image.fromarray(B)
        # apply image transformation
        A = self.transform_A(A)
        B = self.transform_B(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size*64, self.B_size*64)

def downsampling(img, patch_size):
    expo = np.array(img).shape[0].bit_length()
    num_levels = expo - np.array(img).shape[0].bit_length() +1
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    g = gaussian_pyr[-1]
    return g.astype(np.uint8)


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = UnalignedDataset(opt)
    dataset.__getitem__(0)