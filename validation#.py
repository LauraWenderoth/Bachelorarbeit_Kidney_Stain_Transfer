import numpy as np
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.logger import save_images, calculate_evaluation_metrices
from util.logger import Logger
import glob
from util.util import mkdir, tensor2im, save_image
from skimage.metrics import structural_similarity as ssim
from util import util

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

def log_evaluation_metrics(opt, state, wandb_run, model=None, val_dataset=None, visuals=None):
    """log evaluation metrics at W&B
           Parameters:
               opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
               state (String) -- indicates, if metrices should be stored with flag val or train
               model (Model) -- a model to calculate the visuals is needed
               val_Dataset (Dataset) -- a Dataset, that contains all images
               visuals (OrderedDict) - - dictionary of images to display or save
           """
    evaluation_metrics = {'SSMI_A': [], 'SSMI_B': [], 'SSMI_A channel 0': [],
                          'SSMI_A channel 1': [], 'SSMI_A channel 2': [], 'SSMI_B channel 0': [],
                          'SSMI_B channel 1': [],
                          'SSMI_B channel 2': []}
    if val_dataset and model is not None:
        if opt.phase == "val":
            for i, data in enumerate(val_dataset):
                model.set_input(data)
                model.compute_visuals()
                visuals = model.get_current_visuals()
                evaluation_metrics_for_one_image = calculate_evaluation_metrices(visuals, opt)
                for key in evaluation_metrics_for_one_image.keys():
                    evaluation_metrics[key].append(evaluation_metrics_for_one_image[key])
    elif visuals is not None:
        evaluation_metrics_for_one_image = calculate_evaluation_metrices(visuals, opt)
        for key in evaluation_metrics_for_one_image.keys():
            evaluation_metrics[key].append(evaluation_metrics_for_one_image[key])
    else:
        print('No images to calculate evaluation metrics or no model given')

    # log the metrics in weight and biases
    for key in evaluation_metrics.keys():
        if len(evaluation_metrics[key]) > 0:
            metric = np.array(evaluation_metrics[key])
            metric_mean = metric.mean()
            if len(evaluation_metrics[key]) > 1:
                metric_std = metric.std()
                if opt.use_wandb:
                    wandb_run.wandb_run.log(
                        {state + ' ' + key + ' mean': metric_mean, state + ' ' + key + ' std': metric_std})
            else:
                if opt.use_wandb:
                    wandb_run.wandb_run.log({state + ' ' + key : metric_mean})
    return evaluation_metrics

def calculate_evaluation_metrices(visuals, opt):
    """Calculates vor a pair of images (visuals) all evaluation metrices: MSE, SSMI, FID
        Parameters:
            visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        This function will return all evaluation metrices as a dict.
        """
    evaluation_metrics = {}
    number_of_channels = opt.input_nc - 1
    domains = ['A', 'B']
    for domain in domains:
        if ("real_" + domain and "fake_" + domain) in visuals.keys():
            real = visuals["real_" + domain]
            fake = visuals["fake_" + domain]
            real = util.tensor2im(real)
            fake = util.tensor2im(fake)

            # normalise images
            if real.max() > 0:
                real = real / real.max()
            if fake.max() > 0:
                fake = fake / fake.max()

            multichannel = number_of_channels > 1
            ssim_value = ssim(real, fake, gaussian_weights=True, multichannel=multichannel,
                              channel_axis=number_of_channels)
            evaluation_metrics["SSMI_" + domain] = ssim_value
    return evaluation_metrics


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    paths = glob.glob(opt.load_path+"/*net_G_B.pth",recursive=True)

    for path in paths:
        opt.load_path = path

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        logger = Logger(opt)
        # initialize logger
        if opt.use_wandb:
            wandb_run = wandb.init(project='CycleGAN-and-pix2pix-validation', name=opt.name, config=opt,
                                   entity=opt.entity) if not wandb.run else wandb.run
            wandb_run._label(repo='CycleGAN-and-pix2pix-validation')

        if opt.eval:
            model.eval()
        save_path = os.path.join(opt.save_path,opt.name)
        mkdir(save_path)
        number_of_patches = opt.patches_per_width ** 2
        images = {}

        ims_dict = {}

        evaluation_metrics = {'SSMI_A': [], 'SSMI_B': [], 'SSMI_A channel 0': [],
                              'SSMI_A channel 1': [], 'SSMI_A channel 2': [], 'SSMI_B channel 0': [],
                              'SSMI_B channel 1': [],
                              'SSMI_B channel 2':[] }

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
                    save_path_merged = os.path.join(save_path, key)
                    if not os.path.exists(save_path_merged):
                        os.makedirs(save_path_merged)
                    save_image(image, save_path_merged + "/" + image_name + key + "all.png")
                    images[key] = (np.zeros((256 * opt.patches_per_width, 256 * opt.patches_per_width, 3)))
                if opt.use_wandb:
                    wandb.log(ims_dict)

            if patch_index >= opt.num_test * number_of_patches:  # only apply our model to opt.num_test images.
                break

            img_path = model.get_image_paths()  # get image paths
            save_images(save_path, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                        use_wandb=opt.use_wandb)
            evaluation_metrics_for_one_image = log_evaluation_metrics(opt=opt,state="test",wandb_run=wandb_run,visuals=visuals)
            for key in evaluation_metrics_for_one_image.keys():
                evaluation_metrics[key].extend(evaluation_metrics_for_one_image[key])
            if opt.patches_per_width != 1:
                for key in visuals.keys():
                    patch = visuals[key]
                    key = key + " merged"
                    put_image_together(images[key], patch, patch_index, opt.patches_per_width)

        image_name = img_path[0].split("/")[-1]
        image_name = image_name.split(".")[0][:-3]

        for key in images.keys():
            image = images[key]
            image = np.array(image, dtype=np.uint8)
            ims_dict[key] = wandb.Image(image)
            save_path_merged = os.path.join(save_path, key)
            if not os.path.exists(save_path_merged):
                os.makedirs(save_path_merged)
            save_image(image, save_path_merged + "/" + image_name + key + "all.png")
            images[key] = (np.zeros((256 * opt.patches_per_width, 256 * opt.patches_per_width, 3)))
        if opt.use_wandb:
            wandb.log(ims_dict)

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




