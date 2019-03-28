import pandas as pd
import torchvision.datasets
import torch
from imageDataset import imageDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from torchvision import datasets, models, transforms
import time


IMG_SIZE = 32
BATCH_SIZE = 128

train_on_gpu = torch.cuda.is_available() 

dataset = imageDataset('processed_train_data.csv',
              [transforms.ToTensor()])

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

dataloader = [torch.utils.data.DataLoader(x, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4) 
        for x in [train_dataset, valid_dataset]]



       

model_ft = models.resnet18(pretrained = True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
if train_on_gpu:
    model_ft.cuda()


loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_ft.parameters(), lr=0.01 )

start_time = time.time()
if __name__ == '__main__':
    min_valid_loss = np.Inf
    for epoch in range(10):
        train_loss = 0
        valid_loss = 0
        model_ft.train()
        for batch_idx, (img, target) in enumerate(dataloader[0]):
            model_ft.zero_grad()
            if train_on_gpu:
                img, target = img.cuda(), target.cuda()
            target_space = model_ft(img.float())
            loss = loss_function(target_space, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 2 == 0:
                print('trained with ', batch_idx, 'batch,  time cost is', time.time()-start_time)

        
        model_ft.eval()
        for batch_idx, (img, target) in enumerate(dataloader[1]):
            model_ft.zero_grad()
            if train_on_gpu:
                img, target = img.cuda(), target.cuda()
            target_space = model_ft(img.float(0))
            loss = loss_function(target_space, target)
            valid_loss += loss.item()


        
        print('Epoch ', epoch, 'train loss is  ', train_loss, 'valid lostt is', valid_loss)
        if valid_loss < min_valid_loss:
            print('valid loss ',valid_loss,' is les than min valid loss: ', min_valid_loss, ', save model')
            torch.save(model_ft.state_dict(), 'cnnCactusidnetification.pt')
            min_valid_loss = valid_loss

