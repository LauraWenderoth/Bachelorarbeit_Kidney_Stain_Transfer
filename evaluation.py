import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.exposure import histogram


def ssim_score(input1, input2):
    """
    Calculate the mean structural similarity index (SSIM) between two images
    :param input1: Image 1
    :param input2: Image 2 (must have the same shape as Image 1)
    :return: SSIM value
    """
    # RGB images -> multichannel = True
    ssim_value = ssim(input1, input2, gaussian_weights=True, multichannel=True)
    return ssim_value


def psnr_score(ground_truth, test_image):
    """
    Calculate the peak-signal to noise ratio (PSNR) for an image
    :param ground_truth: Ground truth image
    :param test_image: Image to test
    :return: PSNR value in dB
    """
    psnr_value = psnr(ground_truth, test_image)
    return psnr_value


# def vif_score(ground_truth, test_image, p=1):
#     """
#     Apply the pixel based visual information fidelity (vif-p)
#     :param ground_truth: Original input image
#     :param test_image: Transformed input image
#     :param p: Parameter to emphasize large spectral differences
#     :return: vif-p value
#     """
#     p_value = vifp(ground_truth, test_image, p)
#     return p_value


def histogram_wasserstein_distance(hist1, hist2):
    """
    Calculate the wasserstein-distance (earth mover's distance) between two histograms
    :param hist1: Histogram of domain A
    :param hist2: Histogram of domain B
    :return: Distance between distributions
    """
    distance = wasserstein_distance(hist1, hist2)
    return distance





# def get_fid(set_a, set_b):
#     """
#     Calculate the Fréchet Inception Distance (Evaluation metric conceptualised for GANs)
#     :param set_a: Set of images of domain A
#     :param set_b: Set of images of domain B
#     """
#     # Prepare the inception v3 model
#     model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
#     # Prepare images
#     images_a = scale_images(set_a, [299, 299])
#     images_b = scale_images(set_b, [299, 299])
#     images_a = preprocess_input(images_a)
#     images_b = preprocess_input(images_b)
#     # Calculate activations
#     act1 = model.predict(images_a)
#     act2 = model.predict(images_b)
#     # Calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
#     # Calculate sum squared difference between means
#     ssdiff = np.sum((mu1 - mu2) ** 2.0)
#     # Calculate sqrt of product between cov
#     covmean = sqrtm(sigma1.dot(sigma2))
#     # Check and correct imaginary numbers from sqrt
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     # Calculate score
#     score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#     print('FID: %.3f' % score)
#     return score
