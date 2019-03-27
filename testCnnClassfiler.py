import pandas as pd
import torchvision.datasets
import torch
from imageDataset import imageDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np


train_on_gpu = False

dataset = imageDataset('processed_test_data.csv', torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 4) 

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
model.load_state_dict(torch.load('cnnCactusidnetification.pt'))
loss_function = nn.NLLLoss()

model.eval()
test_loss = 0
total = 0
correct = 0
for batch_idx, (img, target) in enumerate(dataloader):
    model.zero_grad()
    if train_on_gpu:
        img, target = img.cuda(), target.cuda()
    target_space = model(img)
    loss = loss_function(target_space, target)
    test_loss += loss.item()
    _, idx = torch.max(target_space, 1)
    prediction = idx.tolist()[0]
    if(prediction == target):
        correct += 1
    total += 1
print("correct accuray is ", correct/total, " ,total:", total, " , correct:", correct)

