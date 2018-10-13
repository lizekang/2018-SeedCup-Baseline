import torch
from DataLoader import MyDataSet
from class_idx import class_idx
from word_idx import word_idx
from config import Config
from model import FC
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from accurancy import F1
import time

opt = Config()
word_with_idx = word_idx(opt)
class_with_idx = class_idx(opt)

print("Loading data...")
trainset = MyDataSet(opt.TRAIN_FILE, word_with_idx, class_with_idx)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.TRAIN_BATCH_SIZE, shuffle=True)
valset = MyDataSet(opt.VAL_FILE, word_with_idx, class_with_idx)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=opt.VAL_BATCH_SIZE, shuffle=False)
print("Loading data successfully!")

net = FC(opt)
if opt.USE_CUDA:
    print('cuda')
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True

criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    net.train()
    print("train epoch:", epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.get_lr(epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if opt.USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        optimizer.zero_grad()
        output_1, output_2, output_3 = net(inputs)
        loss_1 = criterion(output_1, targets[:, 0])
        loss_2 = criterion(output_2, targets[:, 1])
        loss_3 = criterion(output_3, targets[:, 2])
        loss = loss_1+loss_2+loss_3
        loss.backward()
        optimizer.step()
    print("train epoch %d finished" % epoch)


def val(epoch):
    net.eval()
    loss = 0
    f1_1 = F1(opt.NUM_CLASS_1)
    f1_2 = F1(opt.NUM_CLASS_2)
    f1_3 = F1(opt.NUM_CLASS_3)
    for batch_idx, (inputs, targets) in enumerate(valloader):
        if opt.USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        output_1, output_2, output_3 = net(inputs)
        loss_1 = criterion(output_1, targets[:, 0])
        loss_2 = criterion(output_2, targets[:, 1])
        loss_3 = criterion(output_3, targets[:, 2])
        loss += loss_1.item()+loss_2.item()+loss_3.item()
        f1_1.save_data(output_1, targets[:, 0])
        f1_2.save_data(output_2, targets[:, 1])
        f1_3.save_data(output_3, targets[:, 2])
    f1_1 = f1_1.caculate_f1()
    f1_2 = f1_2.caculate_f1()
    f1_3 = f1_3.caculate_f1()
    f1 = 0.1*f1_1+0.3*f1_2+0.6*f1_3
    print("val epoch %d finished" % epoch)
    print("Save model...")
    torch.save(net, "model.t7")
    print("Save model successfully!")
    print("final loss:%.10f" % loss)
    print("f1 for class1:", f1_1)
    print("f1 for class2:", f1_2)
    print("f1 for class3:", f1_3)
    print("final f1:", f1)


for i in range(opt.NUM_EPOCHS):
    train(i)
    val(i)
