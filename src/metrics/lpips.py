import json
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from src.data.data import load_data
from src.models.pix2pix import UnetGenerator

@torch.no_grad()
def calculate_lpips(loader, netG_AB):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  netG_AB = netG_AB.eval().to(device)

  lpips = LearnedPerceptualImagePatchSimilarity('alex').to(device)

  lpips_values = {
    'CycleGAN': {
      'buildings': [],
      'green': [],
      'roads': [],
      'water': []
    },
  }
  for item in tqdm(loader):
    real_A, real_B = item['A'].to(device), item['B'].to(device)

    fake_B = netG_AB(real_A)

    real_B_buildings = real_B.squeeze(0)[0].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    real_B_green = real_B.squeeze(0)[1].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    real_B_roads = real_B.squeeze(0)[2].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    real_B_water = real_B.squeeze(0)[3].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    fake_B_buildings = fake_B.squeeze(0)[0].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    fake_B_green = fake_B.squeeze(0)[1].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    fake_B_roads = fake_B.squeeze(0)[2].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    fake_B_water = fake_B.squeeze(0)[3].unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)


    lpips_values['CycleGAN']['buildings'].append(lpips(real_B_buildings, fake_B_buildings).detach().cpu().numpy())
    lpips_values['CycleGAN']['green'].append(lpips(real_B_green, fake_B_green).detach().cpu().numpy())
    lpips_values['CycleGAN']['roads'].append(lpips(real_B_roads, fake_B_roads).detach().cpu().numpy())
    lpips_values['CycleGAN']['water'].append(lpips(real_B_water, fake_B_water).detach().cpu().numpy())

  for key in lpips_values['CycleGAN'].keys():
    lpips_values['CycleGAN'][key] = str(np.mean(lpips_values['CycleGAN'][key]))

  with open('lpips.json', 'w') as f:
    f.writelines(json.dumps(lpips_values))
    
  return lpips_values

def main():

  _, loader = load_data('data', batch_size=1)
  

  netG_AB = UnetGenerator(input_nc=3, output_nc=4, num_downs=3)
  netG_AB.load_state_dict(torch.load('checkpoints/81/netG_AB.pt'))

  res = calculate_lpips(loader=loader, netG_AB=netG_AB)

  print(res)

if __name__ == '__main__':
  main()
