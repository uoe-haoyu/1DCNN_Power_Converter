import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Build dataset
    """

    def __init__(self, csv_path, transform=None, loader=None, is_val=False, train=True):

        super(MyDataset, self).__init__()
        if csv_path is not None:

            df_train = np.genfromtxt(csv_path, delimiter=',')
        else:
            df_train = None

        self.train = train
        self.df = df_train
        self.transform = transform
        self.loader = loader
        if csv_path is not None:
            print('length_test_dataset：', self.__len__())

    def __getitem__(self, index):


        input = self.df[index, :15]  # 9  1:11
        label = self.df[index, 15:] # 9  13:23

        if self.train:

            flag_noise = random.choice([True, False])
            if flag_noise:
                # # 设置高斯噪声的均值和标准差
                mean = 0
                std_dev = 0.005  # 根据您的数据调整这个值
                # 为前9列添加噪声
                noise = np.random.normal(mean, std_dev, input[:9].shape)
                input[:9] += noise
                # mean = 0

        input = torch.Tensor(input)
        label = torch.Tensor(label)

        return input, label

    def __len__(self):
        if self.df is not None:
            return len(self.df)


def get_data(train_path, test_path, rate=0.1, is_val=False):

    traindata = MyDataset(train_path,train=True)
    testdata = MyDataset(test_path,train=False)

    if is_val:
        valiation = MyDataset(is_val,train=False)

        print(r'训练集占比{}, 将数据集分割为 train:{} test:{} val:{}'.format((testdata.__len__())/(traindata.__len__()+testdata.__len__()+valiation.__len__()), traindata.__len__(),testdata.__len__(), valiation.__len__()))
        return {'train': traindata, 'test': testdata, 'val':valiation}

    else:
        print('没有分割验证集合，只有训练集和测试机')
        return {'train': traindata, 'test': testdata}

def get_pathdata(test_path):
    return MyDataset(test_path,train=False)


if __name__ == '__main__':
    import random


