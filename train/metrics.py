# import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
from torch.nn.modules.loss import _Loss
from scipy.ndimage import gaussian_filter
import numpy
from data.utils import normalize_reverse


def estimate_mask(img):
    mask = img.copy()
    mask[mask > 0.0] = 1.0
    return mask


def mask_pair(x, y, mask):
    return x * mask, y * mask


def im2tensor(image, cent=1., factor=255. / 2.):
    image = image.astype(np.float)
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def psnr_calculate(x, y, val_range=255.0):
    # x,y size (h,w,c)
    # assert len(x.shape) == 3
    # assert len(y.shape) == 3
    x = x.astype(np.float)
    y = y.astype(np.float)
    diff = (x - y) / val_range
    mse = np.mean(diff ** 2)
    psnr = -10 * np.log10(mse)
    return psnr


# def ssim_calculate(x, y, val_range=255.0):
#     ssim = compare_ssim(y, x, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
#                         data_range=val_range)
#     return ssim

def ssim_calculate(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # Processing input image
    img1 = np.array(img1, dtype=np.float32) / 255
    img1 = img1.transpose((2, 0, 1))

    # Processing gt image
    img2 = np.array(img2, dtype=np.float32) / 255
    img2 = img2.transpose((2, 0, 1))


    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return numpy.mean(ssim_map)


# def lpips_calculate(x, y, net='alex', gpu=False):
#     # input range is 0~255
#     # image should be RGB, and normalized to [-1,1]
#     x = im2tensor(x[:, :, ::-1])
#     y = im2tensor(y[:, :, ::-1])
#     loss_fn = lpips.LPIPS(net=net, verbose=False)
#     if gpu:
#         x = x.cuda()
#         y = y.cuda()
#         loss_fn = loss_fn.cuda()
#     lpips_value = loss_fn(x, y)
#     return lpips_value.item()


class PSNR(_Loss):
    def __init__(self, centralize=True, normalize=True, val_range=255.):
        super(PSNR, self).__init__()
        self.centralize = centralize
        self.normalize = normalize
        self.val_range = val_range

    def _quantize(self, img):
        img = normalize_reverse(img, centralize=self.centralize, normalize=self.normalize, val_range=self.val_range)
        img = img.clamp(0, self.val_range).round()
        return img

    def forward(self, x, y):
        diff = self._quantize(x) - self._quantize(y)
        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)

        mse = diff.div(self.val_range).pow(2).view(n, -1).mean(dim=-1)
        psnr = -10 * mse.log10()

        return psnr.mean()
