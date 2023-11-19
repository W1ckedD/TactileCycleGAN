import os
import numpy as np
import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
from skimage.io import imsave

from src.models.pix2pix import UnetGenerator, NLayerDiscriminator
from src.data.data import load_data

class Trainer:
  def __init__(
      self,
      model='pix2pix',
      lr=1e-5,
      betas=(0.5, 0.999),
      lambda_a=10.0,
      lambda_b=10.0,
      epochs=300,
      batch_size=8,
      data_dir='data',
      save_model_every=10,
      save_samples_every=10,
      save_model_dir='checkpoints',
      save_samples_dir='samples'      
    ):
    
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == 'pix2pix':
      self.netG_AB = UnetGenerator(input_nc=3, output_nc=4, num_downs=3).to(self.device)
      self.netG_BA = UnetGenerator(input_nc=4, output_nc=3, num_downs=3).to(self.device)

      self.netD_A = NLayerDiscriminator(input_nc=3).to(self.device)
      self.netD_B = NLayerDiscriminator(input_nc=4).to(self.device)
    else:
      pass

    self.lr = lr
    self.betas = betas
    self.lambda_a = lambda_a
    self.lambda_b = lambda_b
    self.epochs = epochs
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.save_model_every = save_model_every
    self.save_samples_every = save_samples_every
    self.save_model_dir = save_model_dir
    self.save_samples_dir = save_samples_dir

    self.train_loader, _ = load_data(self.data_dir, batch_size=self.batch_size)
    
    self.criterionGAN = nn.BCEWithLogitsLoss().to(self.device)
    self.criterionCycle = nn.L1Loss().to(self.device)

    # Since the number of channels in the source and target data is different, for now we will not use Identity loss.
    # self.criterionIdt = nn.L1Loss() 

    self.optimizer_G = torch.optim.Adam(
      itertools.chain(
        self.netG_AB.parameters(),
        self.netG_BA.parameters()
      ),
      lr=self.lr,
      betas=self.betas
    )

    self.optimizer_D = torch.optim.Adam(
      itertools.chain(
        self.netD_A.parameters(),
        self.netD_B.parameters()
      ),
      lr=self.lr,
      betas=self.betas
    )

  def save_model(self, epoch):
    print('Saving model checkpoints...')
    epoch_path = os.path.join(self.save_model_dir, f"{epoch + 1}")
    os.makedirs(epoch_path, exist_ok=True)

    torch.save(self.netG_AB.state_dict(), f'{epoch_path}/netG_AB.pt')
    torch.save(self.netG_BA.state_dict(), f'{epoch_path}/netG_BA.pt')
    torch.save(self.netD_A.state_dict(), f'{epoch_path}/netD_A.pt')
    torch.save(self.netD_B.state_dict(), f'{epoch_path}/netD_B.pt')

    print('Done...')

  def save_samples(self, epoch, real_A, real_B):
    print('Saving samples')
    epoch_path = os.path.join(self.save_samples_dir, f"{epoch + 1}")
    os.makedirs(epoch_path, exist_ok=True)
    fake_A, fake_B, rec_A, rec_B = self.forward_pass(real_A, real_B)

    real_A = real_A.permute((0, 2, 3, 1)).detach().cpu().numpy()
    real_B = real_B.permute((0, 2, 3, 1)).detach().cpu().numpy()
    fake_A = fake_A.permute((0, 2, 3, 1)).detach().cpu().numpy()
    fake_B = fake_B.permute((0, 2, 3, 1)).detach().cpu().numpy()
    rec_A = rec_A.permute((0, 2, 3, 1)).detach().cpu().numpy()
    rec_B = rec_B.permute((0, 2, 3, 1)).detach().cpu().numpy()

    for i in tqdm(range(real_A.shape[0])):
      sample_path = os.path.join(epoch_path, f"{i}")
      os.makedirs(sample_path, exist_ok=True)
      imsave(f"{sample_path}/real_A.png", real_A[i])
      imsave(f"{sample_path}/fake_A.png", fake_A[i])
      imsave(f"{sample_path}/rec_A.png", rec_A[i])

      np.save(f"{sample_path}/real_B.npy", real_B[i])
      np.save(f"{sample_path}/fake_B.npy", fake_B[i])
      np.save(f"{sample_path}/rec_B.npy", rec_B[i])
    print('Done.')


  def forward_pass(self, real_A, real_B):
    fake_B = self.netG_AB(real_A)
    rec_A = self.netG_BA(fake_B)
    fake_A = self.netG_BA(real_B)
    rec_B = self.netG_AB(fake_A)


    return fake_A, fake_B, rec_A, rec_B

  def train_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
    self.optimizer_G.zero_grad()


    # Backward G
    pred_A = self.netD_A(fake_A)
    loss_G_A = self.criterionGAN(pred_A, torch.tensor(1, device=self.device).expand_as(pred_A).float())
    
    pred_B = self.netD_B(fake_B)
    loss_G_B = self.criterionGAN(pred_B, torch.tensor(1, device=self.device).expand_as(pred_B).float())

    loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_a
    loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_b

    loss_G_adv = loss_G_A + loss_G_B
    loss_cycle =  loss_cycle_A + loss_cycle_B
    loss_G = loss_G_adv + loss_cycle
    loss_G.backward()
    self.optimizer_G.step()

    # return 1, 2
    return loss_G_adv, loss_cycle
  
  def train_D(self, real_A, real_B, fake_A, fake_B):
    self.optimizer_D.zero_grad()

    pred_real_A = self.netD_A(real_A)
    pred_real_B = self.netD_B(real_B)
    pred_fake_A = self.netD_A(fake_A)
    pred_fake_B = self.netD_B(fake_B)

    loss_D_real_A = self.criterionGAN(pred_real_A, torch.tensor(1, device=self.device).expand_as(pred_real_A).float())
    loss_D_real_B = self.criterionGAN(pred_real_B, torch.tensor(1, device=self.device).expand_as(pred_real_B).float())
    loss_D_fake_A = self.criterionGAN(pred_fake_A, torch.tensor(0, device=self.device).expand_as(pred_fake_A).float())
    loss_D_fake_B = self.criterionGAN(pred_fake_B, torch.tensor(0, device=self.device).expand_as(pred_fake_B).float())

    loss_D = 0.25 * (loss_D_real_A + loss_D_real_B + loss_D_fake_A + loss_D_fake_B)
    loss_D.backward()
    
    self.optimizer_D.step()

    return loss_D
  
  def train(self):
    losses_g = []
    losses_cycle = []
    losses_d = []
    
    for i in range(self.epochs):
      print(f'Epoch {i + 1}/{self.epochs}:')

      running_loss_G = 0.0
      running_loss_cycle = 0.0
      running_loss_D = 0.0

      iterator = tqdm(self.train_loader)
      for item in iterator:
        real_A = item['A'].to(self.device)
        real_B = item['B'].to(self.device)
        fake_A, fake_B, rec_A, rec_B = self.forward_pass(real_A, real_B)

        
        batch_loss_G, batch_loss_cycle = self.train_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)
        
        # Train the discriminator every 10 epochs
        if i % 10:
          fake_A, fake_B, rec_A, rec_B = self.forward_pass(real_A, real_B)
          batch_loss_D = self.train_D(real_A, real_B, fake_A, fake_B)
          running_loss_D += batch_loss_D.item()
        msg = f'Lg: {batch_loss_G}, Lcycle: {batch_loss_cycle}, Ld: {batch_loss_D}'
        iterator.set_description((msg))

        running_loss_G += batch_loss_G.item()
        running_loss_cycle += batch_loss_cycle.item()
      
      if i % self.save_model_every == 0:
        self.save_model(i)

      if i % self.save_samples_every == 0:
        with torch.no_grad():
          self.save_samples(i, real_A, real_B)
        

      losses_g.append(running_loss_G / len(iterator))
      losses_cycle.append(running_loss_cycle / len(iterator))
      losses_d.append(running_loss_D / len(iterator))
      
    torch.save(self.netG_AB.state_dict(), 'checkpoints/netG_AB.pt')
    torch.save(self.netG_BA.state_dict(), 'checkpoints/netG_BA.pt')
    torch.save(self.netD_A.state_dict(), 'checkpoints/netD_A.pt')
    torch.save(self.netD_B.state_dict(), 'checkpoints/netD_B.pt')
    return losses_g, losses_cycle, losses_d

if __name__ == '__main__':
  trainer = Trainer(epochs=100)

  l_G, l_C, l_D = trainer.train()

  with open('losses.txt', 'w') as f:
    f.writelines(f"L_G: {l_G}, L_C: {l_C}, L_D: {l_D}")