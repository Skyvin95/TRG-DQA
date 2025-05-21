import os
import scipy
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from resize_center import img_resize
from random_crop_same import resize_crop
from skimage import io, color
from skimage.feature import local_binary_pattern

transform = transforms.ToTensor()
data_path = '/mnt/mdisk/Datasets/DHQ'
matdata = scipy.io.loadmat('/mnt/mdisk/Datasets/DHQ/MOS.mat')
data = {}
data['Haze_name'] = matdata['Haze_name'].flatten()        # hazy images
data['Dehaze_name'] = matdata['Dehaze_name'].flatten()    # dehazed images
data['MOS'] = matdata['MOS'].flatten()

# LBP_params
r = 1
n = 8 * r

class Dataset_train(Dataset):
    def __init__(self):
        self.Haze_imgs = data['Haze_name']
        self.Dehazed_imgs = data['Dehaze_name']
        self.MOS = data['MOS']
        self.transform = transform

    def __len__(self):
        return len(self.Dehazed_imgs)

    def __getitem__(self, idx):
        Haze_img = Image.open(os.path.join(data_path, self.Haze_imgs[idx][0].replace('\\', '/'))).convert('RGB')
        Dehazed_img = Image.open(os.path.join(data_path, self.Dehazed_imgs[idx][0].replace('\\', '/'))).convert('RGB')
        # Haze_img_resized = img_resize(Haze_img)
        # Dehazed_img_resized = img_resize(Dehazed_img)
        Haze_img_resized, Dehazed_img_resized = resize_crop(Haze_img, Dehazed_img)
        Haze_img = self.transform(color.rgb2lab(np.array(Haze_img_resized)))
        Dehazed_img = self.transform(color.rgb2lab(np.array(Dehazed_img_resized)))

        # texture residual map
        Haze_img_G = cv2.cvtColor(np.asarray(Haze_img_resized), cv2.COLOR_RGB2GRAY)
        Haze_lbp = local_binary_pattern(Haze_img_G, n, r)
        Dehazed_img_G = cv2.cvtColor(np.asarray(Dehazed_img_resized), cv2.COLOR_RGB2GRAY)
        Dehazed_lbp = local_binary_pattern(Dehazed_img_G, n, r)
        residual_tex = abs(Dehazed_lbp - Haze_lbp)
        residual_tex = self.transform(residual_tex).cuda()

        # color residual map
        Haze_img_YCbCr = cv2.cvtColor(np.array(Haze_img_resized), cv2.COLOR_RGB2YCrCb)
        Haze_Y, Haze_Cr, Haze_Cb = cv2.split(Haze_img_YCbCr)
        Haze_CBR = Haze_Cb + Haze_Cr
        Dehazed_img_YCbCr = cv2.cvtColor(np.array(Dehazed_img_resized), cv2.COLOR_RGB2YCrCb)
        Dehazed_Y, Dehazed_Cr, Dehazed_Cb = cv2.split(Dehazed_img_YCbCr)
        Dehazed_CBR = Dehazed_Cb + Dehazed_Cr
        residual_color = abs(Dehazed_CBR - Haze_CBR)
        residual_color = self.transform(residual_color).cuda()

        mos = self.MOS[idx]

        return Haze_img, Dehazed_img, residual_tex, residual_color, mos