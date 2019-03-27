import torchvision.datasets
import torch
import pandas as pd

data_frame = pd.read_csv('data/' + 'train.csv')
df0 = data_frame[data_frame.has_cactus ==0]
df1 = data_frame[data_frame.has_cactus ==1][:4364]
df = df0.append(df1, ignore_index=True)
print(len(df0))
print(len(df1))
print(len(df))
df.to_csv('data/processed_data.csv', index=False)