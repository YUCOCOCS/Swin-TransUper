import os

import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob
import random
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        img_ids_png = glob(os.path.join('/home/JianjianYin/project/train_pictures', '*' + '.png'))
        img_ids_png = [os.path.splitext(os.path.basename(p))[0] for p in img_ids_png]
        img_ids_jpg = glob(os.path.join('/home/JianjianYin/project/train_pictures', '*' + '.jpg'))
        img_ids_jpg = [os.path.splitext(os.path.basename(p))[0] for p in img_ids_jpg]
        img_ids = img_ids_png + img_ids_jpg
        random.shuffle(img_ids)
        self.img_ids = img_ids
        self.img_dir = "/home/JianjianYin/project/train_pictures"
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        if 'benign' in  img_id  or 'malignant' in img_id:
            img = cv2.imread(os.path.join(self.img_dir, img_id + ".png"))
            h,w,c = img.shape
            mask = []
            #print(os.path.join(self.mask_dir, img_id + self.mask_ext))
            mask.append(cv2.imread(os.path.join('/home/JianjianYin/project/gt', img_id+"_mask" + '.png'), cv2.IMREAD_GRAYSCALE)[..., None])
            mask = np.dstack(mask)

            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)
            mask = mask.astype('float32') / 255
            mask = mask.transpose(2, 0, 1) # 转换成C*H*W
        
        else:
            img = cv2.imread(os.path.join(self.img_dir, img_id + ".jpg"))
            h,w,c = img.shape
            mask = cv2.imread(os.path.join('/home/JianjianYin/project/gt', img_id + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]
            mask = torch.tensor(mask)
            mask[mask==2]=0 # 其他的类别都置为0
            mask[mask==3]=0
            mask = mask.numpy()
            #if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            img = img.astype('float32')
            img = img.transpose(2, 0, 1)
            mask = mask.astype('float32')
            mask = mask.transpose(2, 0, 1) # 转换成C*H*W
            
            #return img, mask, {'img_id': img_id}
        sample = {'img':img,'mask':mask,'img_id':img_id,'h':h,'w':w}
        return sample



# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
#         """
#         Args:
#             img_ids (list): Image ids.
#             img_dir: Image file directory.
#             mask_dir: Mask file directory.
#             img_ext (str): Image file extension.
#             mask_ext (str): Mask file extension.
#             num_classes (int): Number of classes.
#             transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
#         Note:
#             Make sure to put the files as the following structure:
#             <dataset name>
#             ├── images
#             |   ├── 0a7e06.jpg
#             │   ├── 0aab0a.jpg
#             │   ├── 0b1761.jpg
#             │   ├── ...
#             |
#             └── masks
#                 ├── 0
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 |
#                 ├── 1
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 ...
#         """
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
#         self.num_classes = num_classes
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_ids)

#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
        
#         img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
#         h,w,c = img.shape
#         mask = []
#         print(os.path.join(self.mask_dir, img_id + self.mask_ext))
#         # for i in range(self.num_classes):
#         #     #print(os.path.join(self.mask_dir, str(i),img_id + self.mask_ext))
#         #     mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),img_id +"_mask"+ self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
#         mask.append(cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
#         mask = np.dstack(mask)

#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)
#             img = augmented['image']
#             mask = augmented['mask']
        
#         img = img.astype('float32') / 255
#         img = img.transpose(2, 0, 1)
#         mask = mask.astype('float32') / 255
#         mask = mask.transpose(2, 0, 1) # 转换成C*H*W
        
#         #return img, mask, {'img_id': img_id}
#         sample = {'img':img,'mask':mask,'img_id':img_id,'h':h,'w':w}
#         return sample
