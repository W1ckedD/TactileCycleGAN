import json
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from src.models.pix2pix import UnetGenerator
from src.data.data import load_data
from tqdm import tqdm

@torch.no_grad()
def main():

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  _, loader = load_data('data', batch_size=1)

  netG_AB = UnetGenerator(input_nc=3, output_nc=4, num_downs=3).eval().to(device)
  netG_AB.load_state_dict(torch.load('checkpoints/81/netG_AB.pt'))


  fid_CG_building = FrechetInceptionDistance(feature=64, normalize=True).to(device)
  fid_CG_green = FrechetInceptionDistance(feature=64, normalize=True).to(device)
  fid_CG_roads = FrechetInceptionDistance(feature=64, normalize=True).to(device)
  fid_CG_water = FrechetInceptionDistance(feature=64, normalize=True).to(device)


  for data in tqdm(loader):
    real_A = data['A'].to(device)
    real_B = data['B'].to(device)

    fake_B = netG_AB(real_A)

    real_B_building = real_B[:, 0, :, :].repeat(1, 3, 1, 1)
    real_B_green = real_B[:, 1, :, :].repeat(1, 3, 1, 1)
    real_B_roads = real_B[:, 2, :, :].repeat(1, 3, 1, 1)
    real_B_water = real_B[:, 3, :, :].repeat(1, 3, 1, 1)

    fake_B_building = fake_B[:, 0, :, :].repeat(1, 3, 1, 1)
    fake_B_green = fake_B[:, 1, :, :].repeat(1, 3, 1, 1)
    fake_B_roads = fake_B[:, 2, :, :].repeat(1, 3, 1, 1)
    fake_B_water = fake_B[:, 3, :, :].repeat(1, 3, 1, 1)


    # Cycle GAN
    fid_CG_building.update(real_B_building, real=True)
    fid_CG_building.update(fake_B_building, real=False)

    fid_CG_green.update(real_B_green, real=True)
    fid_CG_green.update(fake_B_green, real=False)
    
    fid_CG_roads.update(real_B_roads, real=True)
    fid_CG_roads.update(fake_B_roads, real=False)

    fid_CG_water.update(real_B_water, real=True)
    fid_CG_water.update(fake_B_water, real=False)

  fid_values = {
    'CycleGAN': {
      'buildings': float(fid_CG_building.compute().detach().cpu().numpy()),
      'green': float(fid_CG_green.compute().detach().cpu().numpy()),
      'roads': float(fid_CG_roads.compute().detach().cpu().numpy()),
      'water': float(fid_CG_water.compute().detach().cpu().numpy()),
    },
  }

  print(fid_values)
  with open('fid.json', 'w') as f:
    f.writelines(json.dumps(fid_values))

  return fid_values


if __name__ == '__main__':
  main()