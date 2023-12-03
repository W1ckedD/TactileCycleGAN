import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.data.data import load_data
from src.models.pix2pix import UnetGenerator

def inference(netG_AB_weights_path, output_path, loader):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  netG_AB = UnetGenerator(input_nc=3, output_nc=4, num_downs=4).eval().to(device)
  netG_AB.load_state_dict(torch.load(netG_AB_weights_path))

  with torch.no_grad():
    for i, item in tqdm(enumerate(loader)):
      save_dir = os.path.join(output_path, f'{i}')
      os.makedirs(save_dir, exist_ok=True)

      real_A = item['A'].to(device)
      real_B = item['B'].to(device)

      fake_B = netG_AB(real_A)

      real_B_building = real_B.squeeze(0)[0].unsqueeze(0)
      real_B_green = real_B.squeeze(0)[1].unsqueeze(0)
      real_B_roads = real_B.squeeze(0)[2].unsqueeze(0)
      real_B_water = real_B.squeeze(0)[3].unsqueeze(0)

      fake_B_building = fake_B.squeeze(0)[0].unsqueeze(0)
      fake_B_green = fake_B.squeeze(0)[1].unsqueeze(0)
      fake_B_roads = fake_B.squeeze(0)[2].unsqueeze(0)
      fake_B_water = fake_B.squeeze(0)[3].unsqueeze(0)

      grid = make_grid([real_B_building, real_B_green, real_B_roads, real_B_water, fake_B_building, fake_B_green, fake_B_roads, fake_B_water], nrow=4)

      save_image(real_A[0], f'{save_dir}/{i}_rgb.png')
      save_image(grid, f'{save_dir}/{i}_grid.png')

if __name__ == '__main__':
  weight_path = 'checkpoints/81/netG_AB.pt'
  _, loader = load_data('data', img_size=(256, 256))
  output_path = 'inference'
  inference(weight_path, output_path, loader)