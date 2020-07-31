from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class mydataset(Dataset):
    def __init__(self,dataroot='',lableroot='',phase='train',shape=224,data_type='dict'):
        self.dataroot = dataroot
        self.lableroot = lableroot
        self.phase = phase
        self.shape = shape
        self.data_type = data_type
        assert phase in ['train', 'val', 'test']
        self.create_lable()

        if phase!='test':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.dataroot,self.image_path[item]+'.jpg')).convert('RGB')
        image = self.transform(image)
        label = self.lable[item]
        return image,label

    def __len__(self):
        return len(self.image_path)

    def create_lable(self):
        if self.data_type == 'dict':
            path = os.path.join(self.lableroot, self.phase + '_mullables.npy')
            dict = np.load(path,allow_pickle=True).item()
            self.image_path = [i for i in dict.keys()]
            self.lable = [i for i in dict.values()]
        else:
            raise ('No this data type')

if __name__ == '__main__':
    dataset = mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/',phase='val')
    print(len(dataset))
    print(dataset[686])

