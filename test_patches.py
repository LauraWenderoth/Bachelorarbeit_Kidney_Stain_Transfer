import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.logger import save_images, calculate_evaluation_metrices

from util.util import mkdir, tensor2im, save_image

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def put_image_together(image, patch, index, patches_per_width):
    patch = tensor2im(patch)
    patch_index = index % patches_per_width ** 2
    x = patch_index // patches_per_width
    y = patch_index % patches_per_width
    patch_size = 256
    image[x * patch_size:x * patch_size + patch_size, y * patch_size:y * patch_size + patch_size] = patch


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix_results', name=opt.name, config=opt,
                               entity=opt.entity) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix_results')

    if opt.eval:
        model.eval()
    save_path = "/home/laurawenderoth/Documents/kidney_microscopy/data/results"
    mkdir(save_path)
    number_of_patches = opt.patches_per_width ** 2
    images = {}
    SSMI_A = []
    SSMI_B = []
    ims_dict = {}
    for patch_index, data in enumerate(dataset):

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        # zusamensetzen der patches
        if patch_index == 0 and opt.patches_per_width != 1:
            for key in visuals.keys():
                key = key + " merged"
                images[key] = (np.zeros((256 * opt.patches_per_width, 256 * opt.patches_per_width, 3)))
        elif patch_index % number_of_patches == 0 and opt.patches_per_width != 1:
            image_name = img_path[0].split("/")[-1]
            image_name = image_name.split(".")[0][:-3]
            for key in images.keys():
                image = images[key]
                image = np.array(image, dtype=np.uint8)
                ims_dict[key] = wandb.Image(image)
                save_image(image, save_path + "/" + image_name + key + "all.png")
                images[key] = (np.zeros((256 * opt.patches_per_width, 256 * opt.patches_per_width, 3)))
            if opt.use_wandb:
                wandb.log(ims_dict)

        if patch_index >= opt.num_test * number_of_patches:  # only apply our model to opt.num_test images.
            break

        img_path = model.get_image_paths()  # get image paths
        save_images(save_path, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                    use_wandb=opt.use_wandb)
        evaluation_metrics = calculate_evaluation_metrices(visuals, opt)
        if opt.patches_per_width != 1:
            for key in visuals.keys():
                patch = visuals[key]
                key = key + " merged"
                put_image_together(images[key], patch, patch_index, opt.patches_per_width)

        if "SSMI_A" in evaluation_metrics.keys():
            SSMI_A.append(evaluation_metrics["SSMI_A"])
            if opt.use_wandb:
                wandb_run.log({'validation SSMI_A': evaluation_metrics["SSMI_A"]})
        if "SSMI_B" in evaluation_metrics.keys():
            SSMI_B.append(evaluation_metrics["SSMI_B"])
            if opt.use_wandb:
                wandb_run.log({'validation SSMI_B': evaluation_metrics["SSMI_B"]})
    if len(SSMI_A) > 0:
        ssmi_a = np.array(SSMI_A)
        ssmi_a = ssmi_a.mean()
        if opt.use_wandb:
            wandb_run.log({'validation SSMI_A mean': ssmi_a})
    if len(SSMI_B) > 0:
        ssmi_b = np.array(SSMI_B)
        ssmi_b = ssmi_b.mean()
        if opt.use_wandb:
            wandb_run.log({'validation SSMI_B mean': ssmi_b})
