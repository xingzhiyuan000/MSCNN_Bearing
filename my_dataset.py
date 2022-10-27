import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        #print(self.images_path[item])
        file=loadmat(self.images_path[item])
        #print(file)
        file_keys = file.keys()
        #print(file_keys)
        for key in file_keys:
            if 'z' in key:
                img = file[key].ravel()
        #print(img.shape)
        label = self.images_class[item]

        if self.transform is not None:
            #img = self.transform(img)
            #print(img)
            img=torch.as_tensor(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        images = images.unsqueeze(1)

        labels = torch.as_tensor(labels)
        return images, labels

