import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UnPairedDataSet(Dataset):
  def __init__(self, source_dir, target_dir, img_size=(256, 256)):
    self.source_dir = source_dir
    self.target_dir = target_dir
    self.img_size = img_size

    self.source_imgs = os.listdir(self.source_dir)
    self.target_imgs = os.listdir(self.target_dir)

    self.transforms = transforms.Compose([
      transforms.ToTensor(),
      # transforms.Resize(self.img_size, antialias=True),
      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  def __getitem__(self, idx):
    path_a = os.path.join(self.source_dir, self.source_imgs[idx])
    path_b = os.path.join(self.target_dir, np.random.choice(self.target_imgs))
    img_a = Image.open(path_a).convert('RGB')
    img_b = np.load(path_b)
    img_b = img_b / 255

    img_a = self.transforms(img_a)
    img_a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_a)
    img_b = self.transforms(img_b)
    img_b = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(img_b)

    return {'A': img_a, 'B': img_b.float()}
  
  def __len__(self):
    return max(len(self.source_imgs), len(self.target_imgs))
  

class PairedDataset(Dataset):
  def __init__(self, source_dir, target_dir, img_size=(224, 224)):
    self.source_dir = source_dir
    self.target_dir = target_dir
    self.img_size = img_size

    self.source_imgs = os.listdir(self.source_dir)
    self.target_imgs = os.listdir(self.target_dir)

    self.transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize(self.img_size, antialias=True),
      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  def __getitem__(self, idx):
    path_a = os.path.join(self.source_dir, self.source_imgs[idx])
    path_b = os.path.join(self.target_dir, self.source_imgs[idx].replace('.png', '.npy'))
    
    img_a = Image.open(path_a)
    img_b = np.load(path_b)

    img_a = self.transforms(img_a)
    img_b = self.transforms(img_b)

    return {'A': img_a, 'B': img_b}

  def __len__(self):
    return len(self.source_imgs)



def load_data(data_dir, batch_size=32, img_size=(256, 256), shuffle=False):
  train_dir = os.path.join(data_dir, 'train')
  val_dir = os.path.join(data_dir, 'val')
  test_dir = os.path.join(data_dir, 'test')

  train_set = UnPairedDataSet(os.path.join(train_dir, 'rgb'), os.path.join(train_dir, 'tactile'), img_size=img_size)

  train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)


  return train_loader, ''
