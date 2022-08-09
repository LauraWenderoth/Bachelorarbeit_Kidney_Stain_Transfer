import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.logger import save_images, calculate_evaluation_metrices
from util.logger import Logger

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
    logger = Logger(opt)
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

    ims_dict = {}

    evaluation_metrics = {'SSMI_A': [], 'SSMI_B': [], 'MSE_B': [], 'MSE_A': [], 'SSMI_A channel 0': [],
                          'SSMI_A channel 1': [], 'SSMI_A channel 2': [], 'SSMI_B channel 0': [],
                          'SSMI_B channel 1': [],
                          'SSMI_B channel 2': [], 'MSE_A channel 0': [], 'MSE_A channel 1': [],
                          'MSE_A channel 2': [],
                          'MSE_B channel 0': [], 'MSE_B channel 1': [], 'MSE_B channel 2': [], 'FID_A': [],
                          'FID_B': []}

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
        evaluation_metrics_for_one_image = logger.log_evaluation_metrics(opt=opt,state="test",visuals=visuals)
        for key in evaluation_metrics_for_one_image.keys():
            evaluation_metrics[key].extend(evaluation_metrics_for_one_image[key])
        if opt.patches_per_width != 1:
            for key in visuals.keys():
                patch = visuals[key]
                key = key + " merged"
                put_image_together(images[key], patch, patch_index, opt.patches_per_width)

    # log the metrics in weight and biases
    for key in evaluation_metrics.keys():
        if len(evaluation_metrics[key]) > 0:
            metric = np.array(evaluation_metrics[key])
            metric_mean = metric.mean()
            if len(evaluation_metrics[key]) > 1:
                metric_std = metric.std()
                if opt.use_wandb:
                    wandb.log(
                        {"test" + ' ' + key + ' mean': metric_mean, 'test' + ' ' + key + ' std': metric_std})
            else:
                if opt.use_wandb:
                    wandb.log({'test' + ' ' + key + ' mean': metric_mean})




