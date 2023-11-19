import os
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imsave
from tqdm import tqdm

from src.data.data import load_data
from src.models.pix2pix import UnetGenerator

class Inference:
  def __init__(
    self,
    netG_AB_weights,
    netG_BA_weights,
  ):
    
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.netG_AB_weights = netG_AB_weights
    self.netG_BA_weights = netG_BA_weights

    self.netG_AB = UnetGenerator(input_nc=3, output_nc=4, num_downs=3).to(self.device)
    self.netG_AB.load_state_dict(torch.load(self.netG_AB_weights))
    self.netG_BA = UnetGenerator(input_nc=4, output_nc=3, num_downs=3).to(self.device)
    self.netG_BA.load_state_dict(torch.load(self.netG_BA_weights))

  def infer(self, sample_path, real_A, real_B):

    fake_B = self.netG_AB(real_A)
    rec_A = self.netG_BA(fake_B)
    fake_A = self.netG_BA(real_B)
    rec_B = self.netG_AB(fake_A)

    real_A = real_A.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy()
    real_B = real_B.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy()
    fake_A = fake_A.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy()
    fake_B = fake_B.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy()
    rec_A = rec_A.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy()
    rec_B = rec_B.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy()

    imsave(f"{sample_path}/real_A.png", real_A)
    imsave(f"{sample_path}/fake_A.png", fake_A)
    imsave(f"{sample_path}/rec_A.png", rec_A)

    np.save(f"{sample_path}/real_B.npy", real_B)
    np.save(f"{sample_path}/fake_B.npy", fake_B)
    np.save(f"{sample_path}/rec_B.npy", rec_B)