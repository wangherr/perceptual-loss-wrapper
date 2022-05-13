import os
import shutil

import torch
from torchvision.datasets.utils import download_url


def download_model(model_name, online_prefix='pretrained'):
    model_name = f'{model_name}.pt'  # add extension
    local_path = f'pretrained/{model_name}'
    if not os.path.isfile(local_path):  # download (only on primary process)
        web_path = f'http://efrosgans.eecs.berkeley.edu/gangealing/{online_prefix}/{model_name}'
        download_url(web_path, 'pretrained')
        local_path = f'pretrained/{model_name}'
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def download_lpips():
    local_path = f'pretrained/lpips_vgg_v0.1.pt'
    if not os.path.isfile(local_path):  # download (only on primary process)
        web_path = 'https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth'
        download_url(web_path, 'pretrained')
        shutil.move('pretrained/vgg.pth', local_path)
