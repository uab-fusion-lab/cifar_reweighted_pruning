import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from testers import *
from vgg import VGG
from resnet import ResNet18, ResNet50
from collections import OrderedDict
import numpy as np


kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_test = transforms.Compose([
    transforms.ToTensor()
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot_heatmap(model):
    for name, weight in model.named_parameters():
        matrix1 = []
        weight = weight.cpu().detach().numpy()
        if len(weight.shape) == 4:
            # for row in range(weight.shape[0]):
            #     temp = []
            #     for column in range(weight.shape[1]):
            #         temp.append(0)
            #     matrix1.append(temp)
            # for row in range(weight.shape[0]):
            #     for column in range(weight.shape[1]):
            #         if np.sum(weight[row, column, :, :]) == 0:
            #             matrix1[row][column] = 0
            #         else:
            #             matrix1[row][column] = 1

            weight2d = weight.reshape(weight.shape[0], -1)
            im = plt.matshow(np.abs(weight2d), cmap=plt.cm.BuPu, aspect='auto')
            plt.colorbar(im)
            plt.title(name)
            # plt.savefig("filter1.png", dpi=800)
            plt.show()

def plot_distribution(model):
    font = {'size': 5}

    plt.rc('font', **font)

    fig = plt.figure(dpi=300)
    i = 1
    for name, weight in model.named_parameters():
        weight = weight.cpu().detach().numpy()
        if len(weight.shape) == 4:
            ax = fig.add_subplot(4, 4, i)
            weight = weight.reshape(1, -1)[0]
            xtick = np.linspace(-0.2, 0.2, 100)
            ax.hist(weight, bins=xtick)
            ax.set_title(name)
            i += 1
    plt.show()

def plot_distribution2(model):
    font = {'size': 5}

    plt.rc('font', **font)

    fig = plt.figure(dpi=300)
    i = 1
    for name, weight in model.named_parameters():
        weight = np.abs(weight.cpu().detach().numpy())
        weight2d = weight.reshape(weight.shape[0], -1)
        column_max = np.max(weight2d, axis=0)
        if len(weight.shape) == 4:
            xtict_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            xx = []
            yy = []
            for j, ticks in enumerate(xtict_values):
                if j == 0:
                    xx.append("<" + str(ticks))
                    yy.append(np.sum(column_max < ticks))
                if j != 0 and j != (len(xtict_values) - 1):
                    xx.append(str(xtict_values[j - 1]) + "~" + str(ticks))
                    yy.append(len(np.where(np.logical_and(column_max >= xtict_values[j - 1], column_max < ticks))[0]))
                if j == (len(xtict_values) - 1):
                    xx.append(">=" + str(ticks))
                    yy.append(np.sum(column_max >= ticks))
            ax = fig.add_subplot(3, 5, i)
            ax.bar(xx, yy, align='center', color="crimson")  # A bar chart
            ax.set_title(name)
            plt.setp(ax, xticks=xx)
            plt.xticks(rotation=90)
            i += 1
    plt.show()


def dataParallel_converter(model, model_path):
    """
        convert between single gpu model and molti-gpu model
    """
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    # model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)
    model.cuda()

    # torch.save(model.state_dict(), './model/cifar10_vgg16_acc_93.540_3fc_sgd_in_multigpu.pt')

    return model

class AttackPGD(nn.Module):
    def __init__(self, basic_model):
        super(AttackPGD, self).__init__()
        self.basic_model = basic_model
        self.rand = True
        self.step_size = 2.0 / 255
        self.epsilon = 8.0 / 255
        self.num_steps = 10

    def forward(self, input, target):  # do forward in the module.py
        # if not args.attack :
        #    return self.basic_model(input), input

        x = input.detach()

        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.cross_entropy(logits, target, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, input - self.epsilon), input + self.epsilon)

            x = torch.clamp(x, 0, 1)

        return self.basic_model(input), self.basic_model(x), x

def main():
    model = ResNet18()
    AttackPGD(model)
    original_model_name = "./model/cifar10_resnet18_adv_acc_48.870_sgd.pt"
    model.load_state_dict(torch.load(original_model_name))

    #model = dataParallel_converter(model, "./model_reweighted/temp3.pt")


    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):
            print(name, weight)
    print("\n------------------------------\n")

    test(model, device, test_loader)

    test_sparsity(model)
    #test_column_sparsity(model)
    # test_filter_sparsity(model)

    # plot_heatmap(model)
    plot_distribution(model)

    current_lr = 0.0001
    rew_milestone = [10, 20, 30, 40]
    xx = np.linspace(1, 400, 400)
    yy = []
    for x in xx:
        if x - 1 in rew_milestone:
            current_lr *= 1.8
        else:
            current_lr *= 0.988
        yy.append(current_lr)
    plt.plot(xx, yy)
    # plt.show()

if __name__ == '__main__':
    main()


