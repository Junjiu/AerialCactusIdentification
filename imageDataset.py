import torchvision.datasets
import torch
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
class imageDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, transform):
        data_frame = pd.read_csv('data/' + text_file)
        self.image_names = data_frame['id'].values
        self.targets = data_frame['has_cactus'].values
        self.transform = transform
        self.minmax = StandardScaler()
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = Image.open('data/train/' + self.image_names[idx])
        img = self.transform(img)
        img = img.reshape(3 * 32 * 32,1)
        img = self.minmax.fit_transform(img)
        img = img.reshape(3, 32, 32)
        target = torch.tensor(self.targets[idx])
        
        return img, target


