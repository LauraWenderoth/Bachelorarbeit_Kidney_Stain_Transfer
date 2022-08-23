import numpy as np
import os
import sys
import ntpath
import time
from . import util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import mean_absolute_error as mae
from util.fid import calculate_fid_given_images

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(save_path, visuals, image_path, aspect_ratio=1.0, use_wandb=True):  # Normal use_wandb=False
    """Save images to the disk.
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = save_path
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        image_dir_label = os.path.join(image_dir, label)
        save_path = os.path.join(image_dir_label, image_name)
        if not os.path.exists(image_dir_label):
            os.makedirs(image_dir_label)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    if use_wandb:
        wandb.log(ims_dict)


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
            mse_value = mse(real, fake)
            evaluation_metrics["MSE_" + domain] = mse_value
            mae_value = mae(real, fake)
            evaluation_metrics["MAE_" + domain] = mae_value
            fid_value = calculate_fid_given_images([[real], [fake]], batch_size=1, device=0, dims=2048)
            evaluation_metrics["FID_" + domain] = fid_value
            # channelwise
            if multichannel:
                for i in range(3):
                    real_channel = real[:, :, i]
                    fake_channel = fake[:, :, i]
                    ssim_value = ssim(real_channel, fake_channel, gaussian_weights=True, multichannel=False)
                    mse_value = mse(real_channel, fake_channel)
                    # fid_value = calculate_fid_given_images([[real_channel], [fake_channel]], batch_size=1, device=0, dims=2048)
                    evaluation_metrics["SSMI_" + domain + " channel " + str(i)] = ssim_value
                    evaluation_metrics["MSE_" + domain + " channel " + str(i)] = mse_value
                    # evaluation_metrics["FID_" + domain + " channel " + str(i)] = fid_value

    return evaluation_metrics


class Logger():
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0

        if self.use_wandb:
            if opt.name == "":
                self.wandb_run = wandb.init(project=opt.project, config=opt,
                                            entity=opt.entity) if not wandb.run else wandb.run
            else:
                self.wandb_run = wandb.init(project=opt.project, name=opt.name, config=opt,
                                            entity=opt.entity) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def save_current_results_wandb(self, visuals, epoch):
        """Display current results on visdom; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

    def plot_current_losses(self, epoch, losses):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if self.use_wandb:
            self.wandb_run.log(losses)
            self.wandb_run.log({'Epoch': epoch})

    def log_evaluation_metrics(self, opt, state, model=None, val_dataset=None, visuals=None):
        """log evaluation metrics at W&B
               Parameters:
                   opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
                   state (String) -- indicates, if metrices should be stored with flag val or train
                   model (Model) -- a model to calculate the visuals is needed
                   val_Dataset (Dataset) -- a Dataset, that contains all images
                   visuals (OrderedDict) - - dictionary of images to display or save
               """
        evaluation_metrics = {'SSMI_A': [], 'SSMI_B': [], 'MSE_B': [], 'MSE_A': [], 'SSMI_A channel 0': [],
                              'SSMI_A channel 1': [], 'SSMI_A channel 2': [], 'SSMI_B channel 0': [],
                              'SSMI_B channel 1': [],
                              'SSMI_B channel 2': [], 'MSE_A channel 0': [], 'MSE_A channel 1': [],
                              'MSE_A channel 2': [],
                              'MAE_B channel 0': [], 'MAE_B channel 1': [], 'MAE_B channel 2': [],'MAE_A channel 0': [], 'MAE_A channel 1': [],
                              'MAE_A channel 2': [],
                              'MSE_B channel 0': [], 'MSE_B channel 1': [], 'MSE_B channel 2': [], 'FID_A': [],
                              'FID_B': []}
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
                    if self.use_wandb:
                        self.wandb_run.log(
                            {state + ' ' + key + ' mean': metric_mean, state + ' ' + key + ' std': metric_std})
                else:
                    if self.use_wandb:
                        self.wandb_run.log({state + ' ' + key : metric_mean})
        return evaluation_metrics

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
