import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from triplet_loss import TripletLoss, select_triplet
import sys

transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)
data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

# 数据装载
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)

# 数据预览
images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)

print(labels)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.35),
            torch.nn.Linear(1024,25)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1,14*14*128)
        x = self.dense(x)
        return x

use_gpu = torch.cuda.is_available()

model = Model()
#torch.save(model, "./model/mymodel.pth")
#model = models.resnet18(pretrained = True)
#fc_features = model.fc.in_features
#model.fc = torch.nn.Linear(fc_features,5)
#model.conv1 = torch.nn.Conv2d(1,8,7,2,3)

#model = torch.load("./model/resnet18.pth")

#cost = torch.nn.CrossEntropyLoss()
cost = TripletLoss()


model = model.cuda()
cost = cost.cuda()

optimizer = torch.optim.Adam(model.parameters())


epoch_n = 5
for epoch in range(epoch_n):
    running_loss = 0.0
    running_correct = 0.0
    time = 0
    print("Epoch{}/{}".format(epoch, epoch_n))
    print("-" * 10)
    for data in data_loader_train:
        x_train, y_train = data
        x_train, y_train = Variable(x_train), Variable(y_train)

        x_train_cuda, y_train_cuda = x_train.cuda(), y_train.cuda()

        embedding_cuda = model(x_train_cuda)
        #print("embedding shape is" + str(embedding_cuda.shape))
        #print(embedding_cuda)
        #print("label is "+str(y_train))
        #tttt = model(x_train_cuda[0])
        #print("Ttttttttt")
        #print(tttt)
        #选择三元组要换回cpu
        embedding_cpu = embedding_cuda.cpu().detach().numpy()
        anchor, pos, neg = select_triplet(embedding_cpu, y_train, 0.2)

        #for x in range(len(anchor)):
            #print(anchor[x], pos[x], neg[x],"++",y_train[anchor[x]], y_train[pos[x]], y_train[neg[x]])
            #print('~'*5)
        if len(anchor)>0:
            anchor_train_cuda = x_train_cuda[anchor+[0]*(8-len(anchor))]
            anchor_embedding = model(anchor_train_cuda)
            pos_train_cuda = x_train_cuda[pos + [0] * (8 - len(pos))]
            pos_embedding = model(pos_train_cuda)
            neg_train_cuda = x_train_cuda[neg + [0] * (8 - len(neg))]
            neg_embedding = model(neg_train_cuda)

            optimizer.zero_grad()
            loss = cost.forward(anchor_embedding, pos_embedding, neg_embedding, 0.2)
            loss = loss.cpu()

            loss.backward()
            optimizer.step()
            #print(str(time)+"****")
            #time += 1
            #print(loss.data)
            running_loss += loss.data
            #running_correct += torch.sum(pred == y_train.data)
            #print(anchor_train_cuda.shape)

            #print(anchor_embedding)
            #anchor_embedding = embedding_cuda[anchor]
            #pos_embedding = embedding_cuda[pos]
            #neg_embedding = embedding_cuda[neg]
            #print(anchor_embedding.shape)
            #print(anchor_embedding)
            #break

    print("loss is "+str(running_loss))

    testing_correct = 0
    for data in data_loader_test:
        x_test, y_test = data
        x_test = Variable(x_test)
        if (use_gpu):
            x_test = x_test.cuda()
        outputs = model(x_test).cpu().detach().numpy()
        dist = np.sum(np.square(outputs - outputs[0]), 1)
        #print(y_test)
        #print(dist)
        threshold = 0.2
        pred = np.zeros(64)
        #print("pred")
        #print(pred)
        xx = np.where(dist < 0.2)
        for i in range(len(xx[0])):
            pred[xx[0][i]] = 1
        #print(pred)
        y_test = np.where(y_test == y_test[0])
        ans = np.zeros(64)
        #print(ans)
        for i in range(len(y_test[0])):
            ans[y_test[0][i]] = 1
        #print(ans)

        testing_correct += np.sum(np.logical_not(np.logical_xor(ans, pred)))
        #print(testing_correct)

    #break
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                      100 * running_correct / len(
                                                                                          data_train),
                                                                                      100 * testing_correct / len(data_test)))

torch.save(model, "./model/model0407.pth")