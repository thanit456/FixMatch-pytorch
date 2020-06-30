import cv2
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.misc import get_mean_and_std

class AnyDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(idx):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.df['filename'].iloc[index])
        image = cv2.imread(img_name)
        
        if self.transform:
            image = self.transform(image)
        return image 

any_dataset = AnyDataset(csv_file='./train.csv', root_dir='./train')
dataloader = DataLoader(any_dataset, batch_size=8, shuffle=True, num_workers=8)
