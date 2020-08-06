# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 09:30:21 2020

@author: ivis
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models 
import pandas as pd
from torch.utils import data
from PIL import Image
from torchvision import transforms

##initialize variable
lr = 1e-03
BatchSize = 4
Epochs18 = 10
Epochs50 = 5
Momentum = 0.9     
Weight_decay = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

y_pred = []

##dataloader
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, transform = None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as rssh ubuntu@140.113.215.195 -p porttc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        ##step1
        path = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(path)
        
        ##step2
        GroundTruth = self.label[index]
        
        ##step3
        img_np = np.asarray(img)/255
        img_np = np.transpose(img_np, (2,0,1))
        img_ten = torch.from_numpy(img_np)
        
        ##step4
        return img_ten, GroundTruth
    def __getlabel__(self):
        return self.label


##ResNet18
class resnet18_sub(nn.Module):
    def __init__(self, insize, outsize, stride):
        super(resnet18_sub, self).__init__()
        self.conv2 = nn.Sequential(
                nn.Conv2d(insize, outsize, kernel_size=3, stride=stride, padding=1, bias=False), 
                nn.BatchNorm2d(outsize),
                nn.ReLU(), 
                nn.Conv2d(outsize, outsize, kernel_size=3, stride=1, padding=1, bias=False), 
                nn.BatchNorm2d(outsize)
                )
        self.downsample = nn.Sequential()
        if stride == 2:
            self.downsample = nn.Sequential(
                    nn.Conv2d(insize, outsize, kernel_size=1, stride=stride, bias=False), 
                    nn.BatchNorm2d(outsize)
                    )
        self.activ = nn.ReLU()
    
    def forward(self, x):
        y = self.conv2(x) + self.downsample(x)
        y = self.activ(y)
        
        return y 

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        ##layer 1
        self.conv2d_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm2d_1 = nn.BatchNorm2d(64)
        self.activ_1 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        ##layer 2
        self.layer1_2 = resnet18_sub(64, 64, 1)
        self.layer2_2 = resnet18_sub(64, 64, 1)
        
        ##layer 3
        self.layer1_3 = resnet18_sub(64, 128, 2)
        self.layer2_3 = resnet18_sub(128, 128, 1)
        
        ##layer 4
        self.layer1_4 = resnet18_sub(128, 256, 2)
        self.layer2_4 = resnet18_sub(256, 256, 1)
        
        ##layer 5
        self.layer1_5 = resnet18_sub(256, 512, 2)
        self.layer2_5 = resnet18_sub(512, 512, 1)
        
        ##layer 6
        self.avgpool2d_6 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.linear_6 = nn.Linear(in_features=51200, out_features=5, bias=True)

    def forward(self, x):
        y = self.conv2d_1(x)
        y = self.batchnorm2d_1(y)
        y = self.activ_1(y)
        y = self.maxpool2d_1(y)
        y = self.layer1_2(y)
        y = self.layer2_2(y)
        y = self.layer1_3(y)
        y = self.layer2_3(y)
        y = self.layer1_4(y)
        y = self.layer2_4(y)
        y = self.layer1_5(y)
        y = self.layer2_5(y)
        y = self.avgpool2d_6(y)
        y = y.view(y.size(0), -1)
        y = self.linear_6(y)
        
        return y


##ResNet50
class resnet50_sub(nn.Module):
    def __init__(self, insize, midsize, outsize, stride, down):
        super(resnet50_sub, self).__init__()
        self.conv2 = nn.Sequential(
                nn.Conv2d(insize, midsize, kernel_size=1, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(midsize),
                nn.ReLU(), 
                nn.Conv2d(midsize, outsize, kernel_size=3, stride=stride, padding=1, bias=False), 
                nn.BatchNorm2d(outsize),
                nn.ReLU(), 
                nn.Conv2d(outsize, outsize, kernel_size=1, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(outsize)
                )
        self.downsample = nn.Sequential()
        if down == True:
            self.downsample = nn.Sequential(
                    nn.Conv2d(insize, outsize, kernel_size=1, stride=stride, bias=False), 
                    nn.BatchNorm2d(outsize)
                    )
        self.activ = nn.ReLU()
    
    def forward(self, x):
        y = self.conv2(x) + self.downsample(x)
        y = self.activ(y)
        
        return y       
        
        
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        
        ##layer 1
        self.conv2d_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm2d_1 = nn.BatchNorm2d(64)
        self.activ_1 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        ##layer 2
        self.layer1_2 = resnet50_sub(64, 64, 256, 1, True)
        self.layer2_2 = resnet50_sub(256, 64, 256, 1, False)
        self.layer3_2 = resnet50_sub(256, 64, 256, 1, False)
        
        ##layer 3
        self.layer1_3 = resnet50_sub(256, 128, 512, 2, True)
        self.layer2_3 = resnet50_sub(512, 128, 512, 1, False)
        self.layer3_3 = resnet50_sub(512, 128, 512, 1, False)
        self.layer4_3 = resnet50_sub(512, 128, 512, 1, False)
        
        ##layer 4
        self.layer1_4 = resnet50_sub(512, 256, 1024, 2, True)
        self.layer2_4 = resnet50_sub(1024, 256, 1024, 1, False)
        self.layer3_4 = resnet50_sub(1024, 256, 1024, 1, False)
        self.layer4_4 = resnet50_sub(1024, 256, 1024, 1, False)
        self.layer5_4 = resnet50_sub(1024, 256, 1024, 1, False)
        self.layer6_4 = resnet50_sub(1024, 256, 1024, 1, False)
        
        ##layer 5
        self.layer1_5 = resnet50_sub(1024, 512, 2048, 2, True)
        self.layer2_5 = resnet50_sub(2048, 512, 2048, 1, False)
        self.layer3_5 = resnet50_sub(2048, 512, 2048, 1, False)
        
        ##layer 6
        self.avgpool2d_6 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.linear_6 = nn.Linear(in_features=204800, out_features=5, bias=True)
        
    def forward(self, x):
        y = self.conv2d_1(x)
        y = self.batchnorm2d_1(y)
        y = self.activ_1(y)
        y = self.maxpool2d_1(y)
        y = self.layer1_2(y)
        y = self.layer2_2(y)
        y = self.layer3_2(y)
        y = self.layer1_3(y)
        y = self.layer2_3(y)
        y = self.layer3_3(y)
        y = self.layer4_3(y)
        y = self.layer1_4(y)
        y = self.layer2_4(y)
        y = self.layer3_4(y)
        y = self.layer4_4(y)
        y = self.layer5_4(y)
        y = self.layer6_4(y)
        y = self.layer1_5(y)
        y = self.layer2_5(y)
        y = self.layer3_5(y)
        y = self.avgpool2d_6(y)
        y = y.view(y.size(0), -1)
        y = self.linear_6(y)
        
        return y

        
def Train(data, model, optimizer):
    ##training
    model.train()
    true = 0
    false = 0
    Loss = nn.CrossEntropyLoss()   #change loss function here
    
    for i, data_ten in enumerate(data):
        train_data, train_label = data_ten
        train_data, train_label = train_data.to(device), train_label.to(device)
            
        x_train = Variable(train_data)
        y_train = Variable(train_label)
            
        prediction = model(x_train.float())
        
        guess = torch.max(prediction, 1)[1]
        for j in range(len(guess)):
            if guess[j] == train_label[j]:
                true = true + 1
            else:
                false = false + 1
        
        if i % 200 == 0:
            print(i, " true: ", true, " false: ", false)
            
        loss = Loss(prediction, y_train.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return true / (true + false)

def Test(data, model, epoch, cur):
    ##testing
    model.eval()
    true = 0
    false = 0
    
    for i, data_ten in enumerate(data):
        test_data, test_label = data_ten
        test_data, test_label = test_data.to(device), test_label.to(device)
            
        x_test = Variable(test_data)
            
        prediction = model(x_test.float())
        
        guess = torch.max(prediction, 1)[1]
        
        if(cur == epoch - 1):
            for j in range(len(guess)):
                y_pred.append(guess[j].item())
             
        for j in range(len(guess)):
            if guess[j] == test_label[j]:
                true = true + 1
            else:
                false = false + 1
        
        if i % 200 == 0:
            print(i, " true: ", true, " false: ", false)
        
    return true / (true + false)

def Pretrained_model18():
    resnet18 = models.resnet18(pretrained = True)
    resnet18.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    
    return resnet18

def Pretrained_model50():
    resnet50 = models.resnet50(pretrained = True)
    resnet50.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
    
    return resnet50

def main():
    transformations = transforms.Compose([transforms.ToTensor()])
    train_data = RetinopathyLoader('data/', 'train', transformations)
    test_data = RetinopathyLoader('data/', 'test', transformations)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BatchSize, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BatchSize, shuffle=False)                                                                                             

    
    """
    ##resnet18
    train_accuracy = []
    test_accuracy = []
    
    #model = ResNet18()
    model = torch.load('resnet18.pkl')
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = Momentum, weight_decay = Weight_decay)
    
    for i in range(Epochs18):
        train = Train(train_dataloader, model, optimizer)
        train_accuracy.append(train)
        test = Test(test_dataloader, model, optimizer, Epochs18, i)
        test_accuracy.append(test)
        print("epochs:", i )
        print('Train Accuracy: ', train)
        print('Test Accuracy: ', test)
    print('Max accuracy: ', max(test_accuracy))
    np.save("train_accuracy_list.npy", train_accuracy)
    np.save("test_accuracy_list.npy", test_accuracy)
    np.save("test_y_pred.npy", y_pred)
    torch.save(model, 'resnet18.pkl')
    #show_AccuracyCurve(train_accuracy, test_accuracy, Epochs18)
    #plot_confusion_matrix(test_dataloader, y_pred)
    """
    
   
    
    ##pretrain resnet18    
    #train_accuracy = []
    test_accuracy = []
    
    #model = Pretrained_model18()
    model = torch.load('resnet18.pkl').to(device)
    """
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = Momentum, weight_decay = Weight_decay)
    y_pred = []
    
    for i in range(Epochs18):
        train = Train(train_dataloader, model, optimizer)
        train_accuracy.append(train)
        test = Test(test_dataloader, model, optimizer, Epochs18, i)
        test_accuracy.append(test)
        print("epochs:", i )
        print('Train Accuracy: ', train)
        print('Test Accuracy: ', test)
    print('Max accuracy: ', max(test_accuracy))
    np.save("pretrain_accuracy_list.npy", train_accuracy)
    np.save("pretest_accuracy_list.npy", test_accuracy)
    np.save("pretest_y_pred.npy", y_pred)
    #torch.save(model, 'preresnet18.pkl')
    """
    
    test = Test(test_dataloader, model, Epochs18, 0)
    print('Test Accuracy: ', test)
    #print('Max accuracy: ', max(test_accuracy))
    #show_AccuracyCurve(train_accuracy, test_accuracy, Epochs18)
    #plot_confusion_matrix(test_dataloader, y_pred)
    #y_true = test_data.__getlabel__()
    #np.save("y_true.npy", y_true)
    

    """
    ##resnet50
    train_accuracy0 = []
    test_accuracy0 = []
    
    model0 = ResNet50()
    model0 = model0.to(device)
    optimizer0 = optim.SGD(model0.parameters(), lr = lr, momentum = Momentum, weight_decay = Weight_decay)
    y_pred = []
    
    for i in range(Epochs50):
        train0 = Train(train_dataloader, model0, optimizer0)
        train_accuracy0.append(train0)
        test0 = Test(test_dataloader, model0, optimizer0, Epochs50, i)
        test_accuracy0.append(test0)
        print("epochs:", i )
        print('Train Accuracy: ', train0)
        print('Test Accuracy: ', test0)
    print('Max accuracy: ', max(test_accuracy0))
    np.save("train_accuracy50_list.npy", train_accuracy0)
    np.save("test_accuracy50_list.npy", test_accuracy0)
    np.save("test_y_pred50.npy", y_pred)
    torch.save(model0, 'resnet50.pkl')
    #show_AccuracyCurve(train_accuracy0, test_accuracy0, Epochs50)
    #plot_confusion_matrix(test_dataloader, y_pred)
    """

    """
    ##pretrain resnet50
    #train_accuracy0 = []
    test_accuracy0 = []
    
    #model0 = Pretrained_model50()
    model0 = torch.load('preresnet50.pkl').to(device)
    
    model0 = model0.to(device)
    optimizer0 = optim.SGD(model0.parameters(), lr = lr, momentum = Momentum, weight_decay = Weight_decay)
    y_pred = []
    
    for i in range(Epochs50):
        train0 = Train(train_dataloader, model0, optimizer0)
        train_accuracy0.append(train0)
        test0 = Test(test_dataloader, model0, optimizer0, Epochs50, i)
        test_accuracy0.append(test0)
        print("epochs:", i )
        print('Train Accuracy: ', train0)
        print('Test Accuracy: ', test0)
    print('Max accuracy: ', max(test_accuracy0))
    np.save("pretrain_accuracy50_list.npy", train_accuracy0)
    np.save("pretest_accuracy50_list.npy", test_accuracy0)
    np.save("pretest_y_pred50.npy", y_pred)
    torch.save(model0, 'preresnet50.pkl')
    #show_AccuracyCurve(train_accuracy0, test_accuracy0, Epochs50)
    #plot_confusion_matrix(test_dataloader, y_pred)
    
    
    test = Test(test_dataloader, model0, Epochs50, 0)
    print('Test Accuracy: ', test)
    #print('Max accuracy: ', max(test_accuracy))
    """
    
    
main()    
