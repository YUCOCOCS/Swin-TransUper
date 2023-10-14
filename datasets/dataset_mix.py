import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.utils.data

class mix_data(torch.utils.data.Dataset):
    def __init__(self,rootpath,transform):
        super().__init__()
        self.root = rootpath
        self.transform = transform
        with open('/home/JianjianYin/transdeeplab/train.txt','r') as f:
            self.ids=f.read().splitlines()
    

    def __getitem__(self, index):
        id = self.ids[index]
        img = cv2.imread(os.path.join(self.root, 'img',id))
        mask = cv2.imread(os.path.join(self.root, 'gt',id.replace(".jpg",".png")), cv2.IMREAD_GRAYSCALE)[..., None]
        mask = torch.tensor(mask)
        mask = mask.numpy()
        #if self.transform is not None:
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1) # 转换成C*H*W
        return img, mask ,id
    
    def __len__(self):
        return len(self.ids)

