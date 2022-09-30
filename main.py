import models.LeNet
import torch.optim as optim
import torch.nn as nn
import torch

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# 将图片数据从 [0,1] 归一化为 [-1, 1] 的取值范围
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def study():
    net = models.LeNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters, lr=0.01)

    input = torch.randn(1, 1, 32, 32)
    # 清空梯度缓存
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(input, output)
    loss.backward()


# 展示图片的函数
def imshow(img):
    img = img / 2 + 0.5  # 非归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    # 随机获取训练集图片
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 展示图片
    imshow(torchvision.utils.make_grid(images))
    # 打印图片类别标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


net = models.LeNet.SimpleNet()

def trainCPU():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start = time.time()
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            # 清空梯度缓存
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            if i % 2000 == 1999:
                # 每 2000 次迭代打印一次信息
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training! Total cost time: ', time.time() - start)


def trainGPU():
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start = time.time()
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 清空梯度缓存
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            if i % 2000 == 1999:
                # 每 2000 次迭代打印一次信息
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training! Total cost time: ', time.time() - start)


if __name__ == '__main__':
    trainGPU()