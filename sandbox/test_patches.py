import os

import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import downsampling,mkdir,tensor2im,save_image

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def put_image_together(image,patch,index,patches_per_width):
    patch = tensor2im(patch)
    patch_index = index % patches_per_width ** 2
    x = patch_index // patches_per_width
    y = patch_index % patches_per_width
    patch_size = 256
    image[x * patch_size:x * patch_size + patch_size, y * patch_size:y * patch_size + patch_size] = patch


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    if opt.eval:
        model.eval()
    save_path = "./results"
    mkdir(save_path)
    number_of_patches = opt.patches_per_width ** 2
    image = np.zeros((256*opt.patches_per_width,256*opt.patches_per_width,3))
    for i, data in enumerate(dataset):
        if i % number_of_patches == 0 and i != 0:
            save_image(np.array(image,dtype=np.uint8), save_path+"/wuhhu.png")
            image = np.zeros((256 * opt.patches_per_width, 256 * opt.patches_per_width, 3))
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(save_path, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                    use_wandb=opt.use_wandb)
        patch = visuals['fake_B']
        put_image_together(image,patch,i,opt.patches_per_width)
    save_image(np.array(image,dtype=np.uint8), save_path + "/wuhhu.png")