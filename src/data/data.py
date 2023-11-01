import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UnPairedDataSet(Dataset):
  def __init__(self, source_dir, target_dir, img_size=(224, 224)):
    self.source_dir = source_dir
    self.target_dir = target_dir
    self.img_size = img_size

    self.source_imgs = os.listdir(self.source_dir)
    self.target_imgs = os.listdir(self.target_dir)

    self.transforms = transforms.Compose([
      transforms.Resize(self.img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  def __getitem__(self, idx):
    path_a = self.source_imgs[idx]
    path_b = np.random.choice(self.target_imgs)
    img_a = Image.open(path_a).conver('RGB')
    img_b = np.load(path_b)

    img_a = self.transforms(img_a)
    img_b = self.transforms(img_b)

    return {'A': img_a, 'B': img_b}
  
  def __len__(self):
    return max(len(self.source_imgs, self.target_imgs))
  

class PairedDataset(Dataset):
  def __init__(self, source_dir, target_dir, img_size=(224, 224)):
    self.source_dir = source_dir
    self.target_dir = target_dir
    self.img_size = img_size

  def __getitem__(self, idx):
    pass

  def __len__(self):
    pass



def load_data(data_dir, batch_size=32):
  train_dir = os.path.join(data_dir, 'train')
  val_dir = os.path.join(data_dir, 'val')
  test_dor = os.path.join(data_dir, 'test')

  train_set = UnPairedDataSet(os.path.join(train_dir, 'rbg'), os.path.join(train_dir, 'tactile'))

  train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2)


  return train_loader