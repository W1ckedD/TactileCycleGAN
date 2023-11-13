import torch
import torch.nn as nn
import itertools
from tqdm import tqdm

from src.models.pix2pix import UnetGenerator, NLayerDiscriminator
from src.data.data import load_data

class Trainer:
  def __init__(self, model='pix2pix', lr=1e-4, betas=(0.5, 0.999), lambda_a=10.0, lambda_b=10.0, epochs=300, batch_size=8, data_dir='data'):
    
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

    self.train_loader, _ = load_data(self.data_dir, batch_size=self.batch_size)
    
    self.criterionGAN = nn.BCEWithLogitsLoss().to(self.device)
    self.criterionCycle = nn.L1Loss().to(self.device)
    # self.criterionIdt = nn.L1Loss() 

    self.optimizer_G = torch.optim.Adam(
      itertools.chain(
        self.netG_A.parameters(),
        self.netG_B.parameters()
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
    loss_G_A = self.criterionGAN(pred_A, torch.tensor(1, device=self.device).expand_as(pred_A))
    
    pred_B = self.netD_B(fake_B)
    loss_G_B = self.criterionGAN(pred_B, torch.tensor(1, device=self.device).expand_as(pred_B))

    loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_a
    loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_b

    loss_G_adv = loss_G_A + loss_G_B
    loss_cycle =  loss_cycle_A + loss_cycle_B
    loss_G = loss_G_adv + loss_cycle
    loss_G.backward()
    self.optimizer_G.step()

    return loss_G_adv, loss_cycle
  
  def train_D(self, real_A, real_B, fake_A, fake_B):
    self.optimizer_D.zero_grad()

    pred_real_A = self.netD_A(real_A)
    pred_real_B = self.netD_B(real_B)
    pred_fake_A = self.netD_A(fake_A)
    pred_fake_B = self.netD_B(fake_B)

    loss_D_real_A = self.criterionGAN(pred_real_A, torch.tensor(1, device=self.device).expand_as(pred_real_A))
    loss_D_real_B = self.criterionGAN(pred_real_B, torch.tensor(1, device=self.device).expand_as(pred_real_B))
    loss_D_fake_A = self.criterionGAN(pred_fake_A, torch.tensor(0, device=self.device).expand_as(pred_fake_A))
    loss_D_fake_B = self.criterionGAN(pred_fake_B, torch.tensor(0, device=self.device).expand_as(pred_fake_B))

    loss_D = loss_D_real_A + loss_D_real_B + loss_D_fake_A + loss_D_fake_B

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
        real_A = item['source'].to(self.device)
        real_B = item['target'].to(self.device)
        fake_A, fake_B, rec_A, rec_B = self.forward_pass(real_A, real_B)

        batch_loss_G, batch_loss_cycle = self.train_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)
        batch_loss_D = self.train_D(real_A, real_B, fake_A, fake_B)

        msg = f'Lg: {batch_loss_G}, Lcycle: {batch_loss_cycle}, Ld: {batch_loss_D}'
        iterator.set_description((msg))

        running_loss_G += batch_loss_G
        running_loss_cycle += batch_loss_cycle
        running_loss_D += batch_loss_D

      losses_g.append(running_loss_G / len(iterator))
      losses_cycle.append(running_loss_cycle / len(iterator))
      losses_d.append(running_loss_D / len(iterator))
      

    return losses_g, losses_cycle, losses_d
