import torch

if __name__ == '__main__':
  device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
  print(device)