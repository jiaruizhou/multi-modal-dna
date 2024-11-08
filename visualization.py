# %% [markdown]
# ## Masked Autoencoders: Visualization Demo
#
# This is a visualization demo using our pre-trained MAE models. No GPU is needed.

# %% [markdown]
# ### Prepare
# Check environment. Install packages if in Colab.
#

# %%
import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import pylab

import models_mae

# %% [markdown]
# ### Define utils

# %%
# define the utils


def show_image(image, title="", path=""):
    # plt.imshow(image, cmap='gray')
    pylab.imshow(image, interpolation='nearest', cmap=cm.gray_r)
    plt.colorbar()  # 添加颜色条，显示灰度值对应的数值范围
    pylab.title(title)

    pylab.savefig(path)
    pylab.close()


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


# %% [markdown]
# ### Load an image

# %%

# %% [markdown]
# ### Load a pre-trained MAE model

# %%
# This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)

chkpt_dir = '/data/zhoujr/AI-Micro/mae/output_dir2/checkpoint-600.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch4')
print('Model loaded.')

# %% [markdown]
# ### Run MAE on the image

# %%
# make random mask reproducible (comment out to make it change)


def run_one_image(img, i, model):
    # x = torch.tensor(img)

    # make it a batch-like
    # x = x.unsqueeze(dim=0)
    # x = torch.einsum('nhwc->nchw', x)

    # run MAE
    x = img.unsqueeze(0)
    print(x.shape)
    loss, y, mask, _ = model(x, mask_ratio=0.75)
    print(y.shape)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 1)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]

    # plt.subplot(1, 4, 1)
    show_image(x[0], "original on data {}".format(i), "./visualize/mask_pictures/{}original.jpg".format(i))

    # plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked on data {}".format(i), "./visualize/mask_pictures/{}masked.jpg".format(i))

    # plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction on data {}".format(i), "./visualize/mask_pictures/{}reconstruction.jpg".format(i))

    # plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible on data {}".format(i), "./visualize/mask_pictures/{}reconstruction_visible.jpg".format(i))


torch.manual_seed(2)
print('MAE with pixel reconstruction:')
import torch

i = 1024
# %%
img = torch.load("./data/final/test_5mer.npy")[i]
print(img.shape)
# show_image(img.squeeze(0), "fcgr of id {}".format(i), "./visualize/mask_pictures/{}_ori.jpg".format(i))
run_one_image(img, i, model_mae)
