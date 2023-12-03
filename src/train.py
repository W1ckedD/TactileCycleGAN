import os
import json
import numpy as np
import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
from torchvision.utils import save_image


from src.models.pix2pix import UnetGenerator, NLayerDiscriminator
from src.data.data import load_data

class Trainer:
  def __init__(
      self,
      model='pix2pix',
      img_size=(256, 256),
      lr=1e-5,
      betas=(0.5, 0.999),
      lambda_a=10.0,
      lambda_b=10.0,
      lambda_idt=0.5,
      epochs=300,
      batch_size=8,
      data_dir='data',
      train_D_every=5,
      save_model_every=10,
      save_samples_every=10,
      save_model_dir='checkpoints',
      save_samples_dir='samples',
      use_idt=False,
      resume_ckpt_dir=None,
      shuffle=False,
    ):
    
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    self.img_size = img_size
    self.lr = lr
    self.betas = betas
    self.lambda_a = lambda_a
    self.lambda_b = lambda_b
    self.lambda_idt = lambda_idt
    self.epochs = epochs
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.train_D_every = train_D_every
    self.save_model_every = save_model_every
    self.save_samples_every = save_samples_every
    self.save_model_dir = save_model_dir
    self.save_samples_dir = save_samples_dir
    self.use_idt = use_idt
    self.resume_ckpt_dir = resume_ckpt_dir
    self.shuffle = shuffle

    if model == 'pix2pix':
      self.netG_AB = UnetGenerator(input_nc=3, output_nc=4, num_downs=4).to(self.device)
      self.netG_BA = UnetGenerator(input_nc=4, output_nc=3, num_downs=4).to(self.device)

      self.netD_A = NLayerDiscriminator(input_nc=3).to(self.device)
      self.netD_B = NLayerDiscriminator(input_nc=4).to(self.device)

      if self.resume_ckpt_dir:
        self.netG_AB.load_state_dict(torch.load(f"{self.resume_ckpt_dir}/netG_AB.pt"))
        self.netG_BA.load_state_dict(torch.load(f"{self.resume_ckpt_dir}/netG_BA.pt"))
        self.netD_A.load_state_dict(torch.load(f"{self.resume_ckpt_dir}/netD_A.pt"))
        self.netD_B.load_state_dict(torch.load(f"{self.resume_ckpt_dir}/netD_B.pt"))
    else:
      pass

    self.train_loader, _ = load_data(self.data_dir, batch_size=self.batch_size, img_size=img_size, shuffle=shuffle)
    
    self.criterionGAN = nn.BCEWithLogitsLoss().to(self.device)
    self.criterionCycle = nn.L1Loss().to(self.device)

    # Since the number of channels in the source and target data is different, for now we will not use Identity loss.
    # OK maybe we should figure out a way to use it after all :)
    self.criterionIdt = nn.L1Loss()

    # Introduce two non-trainable conv layers just to even out the number of channels for samples in each domain
    # Only and only used for Identity loss
    self.AB_nc = nn.Conv2d(3, 4, 1).to(self.device)
    for param in self.AB_nc.parameters():
      param.requires_grad = False

    self.BA_nc = nn.Conv2d(4, 3, 1).to(self.device)
    for param in self.BA_nc.parameters():
      param.requires_grad = False

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
    fake_A, fake_B, rec_A, rec_B, idt_A, idt_B = self.forward_pass(real_A, real_B)


    for i in tqdm(range(real_A.shape[0])):
      sample_path = os.path.join(epoch_path, f"{i}")
      real_path = os.path.join(sample_path, 'real')
      fake_path = os.path.join(sample_path, 'fake')
      rec_path = os.path.join(sample_path, 'rec')

      os.makedirs(real_path, exist_ok=True)
      os.makedirs(fake_path, exist_ok=True)
      os.makedirs(rec_path, exist_ok=True)
      save_image(real_A[i], f"{real_path}/real_A.png")
      save_image(fake_A[i], f"{fake_path}/fake_A.png")
      save_image(rec_A[i], f"{rec_path}/rec_A.png")

      save_image(real_B[i, 0], f"{real_path}/real_B_buildings.png")
      save_image(real_B[i, 1], f"{real_path}/real_B_green.png")
      save_image(real_B[i, 2], f"{real_path}/real_B_roads.png")
      save_image(real_B[i, 3], f"{real_path}/real_B_water.png")

      save_image(fake_B[i, 0], f"{fake_path}/fake_B_buildings.png")
      save_image(fake_B[i, 1], f"{fake_path}/fake_B_green.png")
      save_image(fake_B[i, 2], f"{fake_path}/fake_B_roads.png")
      save_image(fake_B[i, 3], f"{fake_path}/fake_B_water.png")

      save_image(rec_B[i, 0], f"{rec_path}/rec_B_buildings.png")
      save_image(rec_B[i, 1], f"{rec_path}/rec_B_green.png")
      save_image(rec_B[i, 2], f"{rec_path}/rec_B_roads.png")
      save_image(rec_B[i, 3], f"{rec_path}/rec_B_water.png")

    print('Done.')


  def forward_pass(self, real_A, real_B):
    fake_B = self.netG_AB(real_A)
    rec_A = self.netG_BA(fake_B)
    fake_A = self.netG_BA(real_B)
    rec_B = self.netG_AB(fake_A)
    if self.use_idt:
      idt_A = self.netG_AB(self.BA_nc(real_B))
      idt_B = self.netG_BA(self.AB_nc(real_A))
    else:
      idt_A = None
      idt_B = None

    return fake_A, fake_B, rec_A, rec_B, idt_A, idt_B

  def train_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B):
    self.optimizer_G.zero_grad()


    # Backward G
    pred_A = self.netD_A(fake_A)
    loss_G_A = self.criterionGAN(pred_A, torch.tensor(1, device=self.device).expand_as(pred_A).float())
    
    pred_B = self.netD_B(fake_B)
    loss_G_B = self.criterionGAN(pred_B, torch.tensor(1, device=self.device).expand_as(pred_B).float())
    
    loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_a
    loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_b
    if self.use_idt:
      loss_idt_A = self.criterionIdt(real_B, idt_A) * self.lambda_b * self.lambda_idt
      loss_idt_B = self.criterionIdt(real_A, idt_B) * self.lambda_a * self.lambda_idt
    else:
      loss_idt_A = None
      loss_idt_B = None
      

    loss_G_adv = loss_G_A + loss_G_B
    loss_cycle =  loss_cycle_A + loss_cycle_B

    if self.use_idt:
      loss_idt = loss_idt_A + loss_idt_B
    else:
      loss_idt = None

    loss_G = loss_G_adv + loss_cycle
    if self.use_idt:
      loss_G += loss_idt
    loss_G.backward()
    self.optimizer_G.step()


    return loss_G_adv, loss_cycle, loss_idt
  
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
    print(f'Resuming training from checkpint path: {self.resume_ckpt_dir}...')
    print('Please make sure to use a different save path to prevent overwriting the previous checkpoints.')
    user_response = input('DO YOU WISH TO PROCEED?: [yes/no]')
    if user_response.upper() != 'YES':
      print('Training loop stopped.')
      return
    print(f"Training Discriminators every {self.train_D_every} epochs...")
    print(f"Saving model snapshots every {self.save_model_every} epochs...")
    print(f"Saving samples every {self.save_samples_every} epochs...")

    losses_g = []
    losses_cycle = []
    losses_idt = []
    losses_d = []
    
    for i in range(self.epochs):
      print(f'Epoch {i + 1}/{self.epochs}:')

      running_loss_G = 0.0
      running_loss_cycle = 0.0
      running_loss_idt = 0.0
      running_loss_D = 0.0

      iterator = tqdm(self.train_loader)
      for item in iterator:
        real_A = item['A'].to(self.device)
        real_B = item['B'].to(self.device)
        fake_A, fake_B, rec_A, rec_B, idt_A, idt_B = self.forward_pass(real_A, real_B)

        
        batch_loss_G, batch_loss_cycle, batch_loss_idt = self.train_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B)
        msg = f'Lg: {batch_loss_G}, Lcycle: {batch_loss_cycle}{f", Lidt: {batch_loss_idt}" if self.use_idt else ""}'
        
        # Train the discriminator every 5 epochs
        if i % self.train_D_every == 0:
          fake_A, fake_B, rec_A, rec_B, idt_A, idt_B = self.forward_pass(real_A, real_B)
          batch_loss_D = self.train_D(real_A, real_B, fake_A, fake_B)
          running_loss_D += batch_loss_D.item()
          msg += f", Ld: {batch_loss_D}"

        iterator.set_description((msg))

        running_loss_G += batch_loss_G.item()
        running_loss_cycle += batch_loss_cycle.item()
        if self.use_idt:
          running_loss_idt += batch_loss_idt.item()
      
      if i % self.save_model_every == 0:
        self.save_model(i)

      if i % self.save_samples_every == 0:
        with torch.no_grad():
          self.save_samples(i, real_A, real_B)
        

      losses_g.append(running_loss_G / len(iterator))
      losses_cycle.append(running_loss_cycle / len(iterator))
      losses_idt.append(running_loss_idt / len(iterator))
      losses_d.append(running_loss_D / len(iterator))
      
    torch.save(self.netG_AB.state_dict(), 'checkpoints/netG_AB.pt')
    torch.save(self.netG_BA.state_dict(), 'checkpoints/netG_BA.pt')
    torch.save(self.netD_A.state_dict(), 'checkpoints/netD_A.pt')
    torch.save(self.netD_B.state_dict(), 'checkpoints/netD_B.pt')
    return losses_g, losses_cycle, losses_idt, losses_d

if __name__ == '__main__':
  trainer = Trainer(
    img_size=(512, 512),
    epochs=100,
    batch_size=16,
    shuffle=True,
    save_model_every=4,
    save_samples_every=4,
    save_model_dir='checkpoints_512_resume',
    save_samples_dir='samples_512_resume',
    resume_ckpt_dir='checkpoints_512/91'
  )

  l_G, l_C, l_IDT, l_D = trainer.train()


  with open('losses.json', 'w') as f:
    f.writelines(json.dumps({'L_G': l_G, 'L_C': l_C, 'L_IDT': l_IDT, 'L_D': l_D}))