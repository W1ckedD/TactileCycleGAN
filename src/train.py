import torch
import torch.nn as nn
import itertools

from src.models.pix2pix import UnetGenerator, NLayerDiscriminator

class Trainer:
  def __init__(self, model='pix2pix', lr=1e-4, betas=(0.5, 0.999), lambda_a=10.0, lambda_b=10.0):
    
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == 'pix2pix':
      self.netG_A = UnetGenerator(input_nc=3, output_nc=4, num_downs=3).to(self.device)
      self.netG_B = UnetGenerator(input_nc=4, output_nc=3, num_downs=3).to(self.device)

      self.netD_A = NLayerDiscriminator(input_nc=4).to(self.device)
      self.netD_B = NLayerDiscriminator(input_nc=3).to(self.device)
    else:
      pass

    self.lr = lr
    self.betas = betas
    self.lambda_a = lambda_a
    self.lambda_b = lambda_b

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

  def train_G(self, real_A, real_B):
    self.optimizer_G.zero_grad()

    # Forward G
    fake_B = self.netG_A(real_A)
    rec_A = self.netG_B(fake_B)
    fake_A = self.netG_B(real_B)
    rec_B = self.netG_A(fake_A)

    # Backward G
    with torch.no_grad():
      pred_A = self.netD_A(fake_B)
      loss_G_A = self.criterionGAN(pred_A, torch.tensor(1, device=self.device).expand_as(pred_A))
    
      pred_B = self.netD_B(fake_A)
      loss_G_B = self.criterionGAN(pred_B, torch.tensor(1, device=self.device).expand_as(pred_B))

    loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_a
    loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_b

    loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
    loss_G.backward()
    self.optimizer_G.step()

    return loss_G
  
  def train_D(self):
    pass

