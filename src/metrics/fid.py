"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse
import json

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
# from core.data_loader import get_eval_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=50):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value

@torch.no_grad()
def c_fid(loader, generator):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  inception = InceptionV3().eval().to(device)
  generator = generator.eveal().to(device)

  actvs = {
    'real_building': [],
    'real_green': [],
    'real_roads': [],
    'real_water': [],
    'fake_building': [],
    'fake_green': [],
    'fake_roads': [],
    'fake_water': [],
  }

  mu = {
    'real_building': 0,
    'real_green': 0,
    'real_roads': 0,
    'real_water': 0,
    'fake_building': 0,
    'fake_green': 0,
    'fake_roads': 0,
    'fake_water': 0,
  }

  cov = {
    'real_building': 0,
    'real_green': 0,
    'real_roads': 0,
    'real_water': 0,
    'fake_building': 0,
    'fake_green': 0,
    'fake_roads': 0,
    'fake_water': 0,
  }


  for data in loader:
    real_A = data['A'].to(device)
    real_B = data['B'].squeeze(0).to(device)

    fake_B = generator(real_A)


    real_B_building = real_B[0].repeat(3, 1, 1).unszueeze(0)
    real_B_green = real_B[1].repeat(3, 1, 1).unszueeze(0)
    real_B_roads = real_B[2].repeat(3, 1, 1).unszueeze(0)
    real_B_water = real_B[3].repeat(3, 1, 1).unszueeze(0)

    fake_B_building = fake_B[0].repeat(3, 1, 1).unszueeze(0)
    fake_B_green = fake_B[1].repeat(3, 1, 1).unszueeze(0)
    fake_B_roads = fake_B[2].repeat(3, 1, 1).unszueeze(0)
    fake_B_water = fake_B[3].repeat(3, 1, 1).unszueeze(0)


    # Calculating actv of real channels
    actv_real_building = inception(real_B_building)
    actvs['real_building'].append(actv_real_building)

    actv_real_green = inception(real_B_green)
    actvs['real_green'].append(actv_real_green)

    actv_real_roads = inception(real_B_roads)
    actvs['real_roads'].append(actv_real_roads)

    actv_real_water = inception(real_B_water)
    actvs['real_water'].append(actv_real_water)

    # Calculating actv of generated channels
    actv_fake_building = inception(fake_B_building)
    actvs['fake_building'].append(actv_fake_building)

    actv_fake_green = inception(fake_B_green)
    actvs['fake_green'].append(actv_fake_green)

    actv_fake_roads = inception(fake_B_roads)
    actvs['fake_roads'].append(actv_fake_roads)

    actv_fake_water = inception(fake_B_water)
    actvs['fake_water'].append(actv_fake_water)
  
  actvs['real_building'] = torch.cat(actvs['real_building'], dim=0).cpu().detach().numpy()
  actvs['real_green'] = torch.cat(actvs['real_green'], dim=0).cpu().detach().numpy()
  actvs['real_roads'] = torch.cat(actvs['real_roads'], dim=0).cpu().detach().numpy()
  actvs['real_water'] = torch.cat(actvs['real_water'], dim=0).cpu().detach().numpy()

  actvs['fake_building'] = torch.cat(actvs['fake_building'], dim=0).cpu().detach().numpy()
  actvs['fake_green'] = torch.cat(actvs['fake_green'], dim=0).cpu().detach().numpy()
  actvs['fake_roads'] = torch.cat(actvs['fake_roads'], dim=0).cpu().detach().numpy()
  actvs['fake_water'] = torch.cat(actvs['fake_water'], dim=0).cpu().detach().numpy()

  mu['real_building'] = np.mean(actvs['real_building'], axis=0)
  mu['real_green'] = np.mean(actvs['real_green'], axis=0)
  mu['real_roads'] = np.mean(actvs['real_roads'], axis=0)
  mu['real_water'] = np.mean(actvs['real_water'], axis=0)

  mu['fake_building'] = np.mean(actvs['fake_building'], axis=0)
  mu['fake_green'] = np.mean(actvs['fake_green'], axis=0)
  mu['fake_roads'] = np.mean(actvs['fake_roads'], axis=0)
  mu['fake_water'] = np.mean(actvs['fake_water'], axis=0)

  cov['real_building'] = np.cov(actvs['real_building'], rowvar=False)
  cov['real_green'] = np.cov(actvs['real_green'], rowvar=False)
  cov['real_roads'] = np.cov(actvs['real_roads'], rowvar=False)
  cov['real_water'] = np.cov(actvs['real_water'], rowvar=False)

  cov['fake_building'] = np.cov(actvs['fake_building'], rowvar=False)
  cov['fake_green'] = np.cov(actvs['fake_green'], rowvar=False)
  cov['fake_roads'] = np.cov(actvs['fake_roads'], rowvar=False)
  cov['fake_water'] = np.cov(actvs['fake_water'], rowvar=False)

  fid = {
    'building': frechet_distance(mu['real_building'], cov['real_building'], mu['fake_building'], cov['fake_building']),
    'green': frechet_distance(mu['real_green'], cov['real_green'], mu['fake_green'], cov['fake_green']),
    'roads': frechet_distance(mu['real_roads'], cov['real_roads'], mu['fake_roads'], cov['fake_roads']),
    'water': frechet_distance(mu['real_water'], cov['real_water'], mu['fake_water'], cov['fake_water']),
  }

  with open('fid.json', 'w') as f:
    f.writelines(json.dumps(fid))

  return fid
  


  



if __name__ == '__main__':
  loader = None
  G_AB = None
  fid = c_fid(loader)

# python -m metrics.fid --paths PATH_REAL PATH_FAKE