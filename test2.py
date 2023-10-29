import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import scipy.misc
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



data_path = 'wineQuality/data/'
data_name = ['winequality-red.csv', 'winequality-white.csv']
'''十一个参数, 一个目标值'''

def load_data(path=data_path, name=data_name[0]):
    print('loading data:')
    print('loading ' + name)
    df = pd.read_csv(path+name,sep=';')
    arr = np.array(df)
    dataList = arr.tolist()
    print(len(dataList))
    return dataList

def z_score(data):
    print('z_score:')
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data-mean)/std
    data = data.tolist()
    print(data[0])
    return data

def create_dataset(path=data_path, name=data_name[0]):
    print('creating dataset:')
    dataList = load_data(path, name) 
    print(dataList[0])
    x = [data[:-1] for data in dataList]
    y = [data[-1] for data in dataList]
    for i in range(len(y)):
        y[i] = int(y[i])-3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    x_train = z_score(x_train)
    x_test = z_score(x_test)
    print(len(x_train), len(x_test))
    return x_train, y_train, x_test, y_test, len(x_train), len(x_test)

def change_data_type(data):
    print('changing data type to numpy array')
    data = np.asarray(data)
    return data

class WineDataset(Dataset):
    def __init__(self, data, label):
        super(WineDataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self,index):
        data = torch.Tensor(self.data[index])
        label = torch.IntTensor(self.label[index])
        return data, label

    def __len__(self):
        return len(self.label)

def wineDataLoader(length, test_rate, Resize, batch_size, name=data_name[0]):
    print('constructing data loader')
    train_data_self, train_data_label, test_data_self, test_data_label, train_len, test_len = create_dataset(name=name)
    train_data_self = change_data_type(train_data_self)
    for i in range(len(train_data_label)):
        train_data_label[i] = [train_data_label[i]]
    train_data_label = change_data_type(train_data_label)
    test_data_self = change_data_type(test_data_self)
    for i in range(len(test_data_label)):
        test_data_label[i] = [test_data_label[i]]
    test_data_label = change_data_type(test_data_label)
    train_data = WineDataset(train_data_self, train_data_label)
    test_data = WineDataset(test_data_self, test_data_label)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_loader, test_loader, train_len, test_len

'''待写'''
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=11,out_features=22,bias=True)
        self.linear2 = nn.Linear(in_features=22,out_features=23,bias=True)
        self.linear3 = nn.Linear(23, 7)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

def train(train_loader, train_len, epoches=50):
    # 学习率0.001
    learning_rate = 1e-3
    batch_size = 100

    lenet = LeNet()

    criterian = nn.CrossEntropyLoss(reduction='sum')  # loss（损失函数）
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)  # optimizer（迭代器）

    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for (data, label) in train_loader:  # 从train_loader中每次取出batch_size=100的data和label
            '''将label数据结构由[[1],[2],[3]]转换为[1,2,3]'''
            label = label.squeeze(1)
            label = label.type(torch.LongTensor)
            optimizer.zero_grad()  # 求梯度之前对梯度清零以防梯度累加
            output=lenet(data)  # 对模型进行前向推理
            '''Target 5 is out of bounds.'''
            loss=criterian(output,label)  # 计算本轮推理的Loss值
            loss.backward()    # loss反传存到相应的变量结构当中
            optimizer.step()   # 使用计算好的梯度对参数进行更新
            running_loss+=loss.item()
            _,predict=torch.max(output,1)  # 计算本轮推理的准确率
            correct_num=(predict==label).sum()
            running_acc+=correct_num.item()
 
        running_loss/=train_len
        running_acc/=train_len
        plt.plot(i, running_loss, 'b.')
        print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, epoches, running_loss,100 * running_acc))

    plt.show()

    return lenet

def test(test_loader, test_len, lenet):
    lenet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, label) in test_loader:
            label = label.squeeze(1)
            label = label.type(torch.LongTensor)
            output = lenet(data)
            _,predict=torch.max(output,1)
            correct += (predict == label).sum().item()
            total += label.size(0)
    print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    train_loader, test_loader, train_len, test_len = wineDataLoader(100, 0.25, 0.5, 100, name=data_name[0])


    lenet = train(train_loader, train_len, 1200)
    test(test_loader, test_len, lenet)
    


    lenet = train(train_loader, train_len)
    
