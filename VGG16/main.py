import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

from dataset import trainloader, testloader, classes
from model import VGG16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VGG16()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if torch.cuda.is_available():
    print("Using GPU: ",torch.cuda.get_device_name(0))
    model.cuda()
    gpu = 1

num_epoch = 100


train_loss = []
test_loss = []
train_accu = []
test_accu = []


for epoch in range(num_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print("Epoch ", epoch, " training loss: ", running_loss / len(trainloader ))
    print(f'Accuracy: {(100 * correct / total)}%')
    train_loss.append( running_loss / len(trainloader))
    train_accu.append( 100 * correct / total )

    running_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        running_loss += loss.item()
    
    print("Epoch ", epoch, " testing loss: ", running_loss / len(testloader))
    print(f'Accuracy: {(100 * correct / total)}%')
    test_loss.append( running_loss / len(testloader))
    test_accu.append( 100 * correct / total )
    
    np.savetxt("train_loss.csv", np.asarray( train_loss ), delimiter=",")
    np.savetxt("test_loss.csv", np.asarray( test_loss ), delimiter=",")
    np.savetxt("train_accu.csv", np.asarray( train_accu ), delimiter=",")
    np.savetxt("test_accu.csv", np.asarray( test_accu ), delimiter=",")

print('Finished Training')

"""
dataiter = iter(testloader)
images, labels = dataiter.next()
"""

torch.save(model.state_dict(), 'VGG16_epoch100.pt')
