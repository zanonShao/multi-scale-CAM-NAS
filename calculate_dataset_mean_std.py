from MyDataloader import mydataset
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/'
phase='train'
path = os.path.join(lableroot, phase + '_mullables.npy')
# dict = np.load(path,allow_pickle=True).item()
a = mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
                  lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='train')
b = DataLoader(a)

# 输入PyTorch的dataset，输出均值和标准差
mean_r = 0
mean_g = 0
mean_b = 0
print("计算均值>>>")
dataset = b
for (img, _)in tqdm(dataset,ncols=80):
  # img=Image.open(img_path)
  # img = np.asarray(img) # change PIL Image to numpy array
  img = img.numpy().squeeze()
  mean_b += np.mean(img[0, :, :])
  mean_g += np.mean(img[1, :, :])
  mean_r += np.mean(img[2, :, :])

mean_b /= len(dataset)
mean_g /= len(dataset)
mean_r /= len(dataset)

diff_r = 0
diff_g = 0
diff_b = 0

N = 0
print("计算方差>>>")
for (img, _)in tqdm(dataset,ncols=80):
  # img=Image.open(img_path)
  img = img.numpy().squeeze()
  diff_b += np.sum(np.power(img[0, :, :] - mean_b, 2))
  diff_g += np.sum(np.power(img[1, :, :] - mean_g, 2))
  diff_r += np.sum(np.power(img[2, :, :] - mean_r, 2))

  N += np.prod(img[0, :, :].shape)

std_b = np.sqrt(diff_b / N)
std_g = np.sqrt(diff_g / N)
std_r = np.sqrt(diff_r / N)

mean = (mean_b.item(), mean_g.item(), mean_r.item())
std = (std_b.item(), std_g.item(), std_r.item())

val_mean,val_std=mean,std
#print("训练集的平均值：{}，方差：{}".format(train_mean,train_std))
print("验证集的平均值：{}".format(val_mean))
print("验证集的方差：{}".format(val_mean))
#print("测试集的平均值：{}，方差：{}".format(test_mean,test_std))