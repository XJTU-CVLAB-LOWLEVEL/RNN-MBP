import random
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
# from detectron2.layers import ModulatedDeformConv, DeformConv


# class ModulatedDeformLayer(nn.Module):
#     """
#     Modulated Deformable Convolution (v2)
#     """
#
#     def __init__(self, in_chs, out_chs, kernel_size=3, deformable_groups=1, activation='relu'):
#         super(ModulatedDeformLayer, self).__init__()
#         assert isinstance(kernel_size, (int, list, tuple))
#         self.deform_offset = conv3x3(in_chs, (3 * kernel_size ** 2) * deformable_groups)
#         self.act = actFunc(activation)
#         self.deform = ModulatedDeformConv(
#             in_chs,
#             out_chs,
#             kernel_size,
#             stride=1,
#             padding=1,
#             deformable_groups=deformable_groups
#         )
#
#     def forward(self, x, feat):
#         offset_mask = self.deform_offset(feat)
#         offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
#         offset = torch.cat((offset_x, offset_y), dim=1)
#         mask = mask.sigmoid()
#         out = self.deform(x, offset, mask)
#         out = self.act(out)
#         return out
#
#
# class DeformLayer(nn.Module):
#     """
#     Deformable Convolution (v1)
#     """
#
#     def __init__(self, in_chs, out_chs, kernel_size=3, deformable_groups=1, activation='relu'):
#         super(DeformLayer, self).__init__()
#         self.deform_offset = conv3x3(in_chs, (2 * kernel_size ** 2) * deformable_groups)
#         self.act = actFunc(activation)
#         self.deform = DeformConv(
#             in_chs,
#             out_chs,
#             kernel_size,
#             stride=1,
#             padding=1,
#             deformable_groups=deformable_groups
#         )
#
#     def forward(self, x, feat):
#         offset = self.deform_offset(feat)
#         out = self.deform(x, offset)
#         out = self.act(out)
#         return out


def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img

    return [_augment(a) for a in args]


def postprocess(*images, rgb_range, ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        if ycbcr_flag:
            out = img.mul(rgb_coefficient).clamp(16, 235)
        else:
            out = img.mul(rgb_coefficient).clamp(0, 255).round()

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]


def calc_psnr(img1, img2, rgb_range=1., shave=4):
    if isinstance(img1, torch.Tensor):
        img1 = img1[:, :, shave:-shave, shave:-shave]
        img1 = img1.to('cpu').numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2[:, :, shave:-shave, shave:-shave]
        img2 = img2.to('cpu').numpy()
    mse = np.mean((img1 / rgb_range - img2 / rgb_range) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_grad_sobel(img, device='cuda'):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = torch.mean(img, dim=1, keepdim=True)

    # img = calc_meanFilter(img, device=device)  # meanFilter

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = torch.from_numpy(sobel_filter_X).float().to(device)
    sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().to(device)
    grad_X = F.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_meanFilter(img, kernel_size=11, n_channel=1, device='cuda'):
    mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    mean_filter_X = torch.from_numpy(mean_filter_X).float().to(device)
    new_img = torch.zeros_like(img)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                             stride=1, padding=kernel_size // 2)
    return new_img
