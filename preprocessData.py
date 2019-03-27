import torchvision.datasets
import torch
import pandas as pd
import sklearn.utils

data_frame = pd.read_csv('data/' + 'train.csv')
df0 = data_frame[data_frame.has_cactus ==0]
df1 = data_frame[data_frame.has_cactus ==1][:4364]
df = df0.append(df1, ignore_index=True)
print(len(df0))
print(len(df1))
print(len(df))
df = sklearn.utils.shuffle(df)
print(df)
train_size = int(0.8*len(df))
df[:train_size].to_csv('data/processed_train_data.csv', index=False)
df[train_size:].to_csv('data/processed_test_data.csv', index=False)