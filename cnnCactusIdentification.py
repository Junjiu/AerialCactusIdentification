import pandas as pd
import torchvision.datasets
import torch
from imageDataset import imageDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np


IMG_SIZE = 32
BATCH_SIZE = 5

train_on_gpu = False 

dataset = imageDataset('processed_train_data.csv', torchvision.transforms.ToTensor())

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

dataloader = [torch.utils.data.DataLoader(x, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4) 
        for x in [train_dataset, valid_dataset]]


class cnnCactusIdentification(nn.Module):
    def __init__(self):
        super(cnnCactusIdentification, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(3, 64, 2)
        self.pool_layer1 = nn.MaxPool2d(3)

        self.conv_layer2 = torch.nn.Conv2d(64, 32, 2)
        self.pool_layer2 = nn.MaxPool2d(3)

        self.fc_layer1 = nn.Linear(288, 64)
        self.fc_layer2 = nn.Linear(64, 2)

    def forward(self, img):
        conv1 = F.relu(self.conv_layer1(img.float()))
        pool1 = self.pool_layer1(conv1)

        conv2 = F.relu(self.conv_layer2(pool1))
        pool2 = self.pool_layer2(conv2)
        fc1 = self.fc_layer1(pool2.view(img.size(0),-1))
        fc2 = self.fc_layer2(fc1)
        return F.log_softmax(fc2, dim = 1)
       

model = cnnCactusIdentification()
if train_on_gpu:
    model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    min_valid_loss = np.Inf
    for epoch in range(10):
        train_loss = 0
        valid_loss = 0
        model.train()
        for batch_idx, (img, target) in enumerate(dataloader[0]):
            model.zero_grad()
            if train_on_gpu:
                img, target = img.cuda(), target.cuda()
            target_space = model(img)
            loss = loss_function(target_space, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        
        model.eval()
        for batch_idx, (img, target) in enumerate(dataloader[1]):
            model.zero_grad()
            if train_on_gpu:
                img, target = img.cuda(), target.cuda()
            target_space = model(img)
            loss = loss_function(target_space, target)
            valid_loss += loss.item()


        
        print('Epoch ', epoch, 'train loss is  ', train_loss, 'valid lostt is', valid_loss)
        if valid_loss < min_valid_loss:
            print('valid loss ',valid_loss,' is les than min valid loss: ', min_valid_loss, ', save model')
            torch.save(model.state_dict(), 'cnnCactusidnetification.pt')
            min_valid_loss = valid_loss

