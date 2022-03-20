import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize, Resize, ColorJitter
import numpy as np
from torch.utils.data import DataLoader
from dataset import PixWiseDataset
from model import DeePixBiS
from loss import PixWiseBCELoss
from metrics import predict, test_accuracy, test_loss
from trainer import Trainer

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device} for training')
model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS_compose.pth'))
model.eval()
loss_fn = PixWiseBCELoss()

opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001 )

train_tfms = Compose([Resize([224, 224]),
                      RandomHorizontalFlip(0.5),
                      ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), 
                      ToTensor(),
                      Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_tfms = Compose([Resize([224, 224]),
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = PixWiseDataset('./train_compose.csv', transform=train_tfms)
train_ds = train_dataset.dataset()

val_dataset = PixWiseDataset('./test_compose.csv', transform=test_tfms)
val_ds = val_dataset.dataset()

batch_size = 4
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

for x, y, z in val_dl:
	_, zp = model(x)
	print(zp)
	print (z)
	break

print(test_accuracy(model.to(device), train_dl))
print(test_loss(model.to(device), train_dl, loss_fn))

# 5 epochs ran

# trainer = Trainer(train_dl, val_dl, model, 10, opt, loss_fn,device)

# print('Training Beginning\n')
# trainer.fit()
# print('\nTraining Complete')
# torch.save(model.state_dict(), './DeePixBiS_compose.pth')