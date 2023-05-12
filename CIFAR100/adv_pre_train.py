import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

import torchvision
import torchvision.transforms as transforms

# from lenet import LeNet
from mobilenet import MobileNet
from vgg import VGG
# from convnet import ConvNet
from resnet import ResNet18, ResNet50

import prune_util
from prune_util import GradualWarmupScheduler
from prune_util import CrossEntropyLossMaybeSmooth
from prune_util import mixup_data, mixup_criterion

# from regularizer import L1Regularizer, L2Regularizer, ElasticNetRegularizer, GroupSparseLassoRegularizer, GroupLassoRegularizer




# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
parser.add_argument('--epsilon', type=float, default=8.0,
                    help='default is 8.0')
parser.add_argument('--step-size', type=float, default=2.0,
                    help='default is 2.0')
parser.add_argument('--num-steps', default=10, type=int,
                    help='default is 10')
parser.add_argument('--random-start', action='store_true', default=True,
                    help="adv random")



class AttackPGD(nn.Module):
    def __init__(self, basic_model):
        super(AttackPGD, self).__init__()
        self.basic_model = basic_model
        self.rand = args.random_start
        self.step_size = args.step_size / 255
        self.epsilon = args.epsilon / 255
        self.num_steps = args.num_steps

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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if not os.path.exists(args.save):
    os.makedirs(args.save)

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#
# train_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR10('./data.cifar10', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.Pad(4),
#                            transforms.RandomCrop(32),
#                            transforms.RandomHorizontalFlip(),
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                        ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                        ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()

])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

if args.cuda:
    if args.arch == "vgg":
        if args.depth == 16:
            model = VGG(depth=16, init_weights=True, cfg=None)
        elif args.depth == 19:
            model = VGG(depth=19, init_weights=True, cfg=None)
        else:
            sys.exit("vgg doesn't have those depth!")
    elif args.arch == "resnet":
        if args.depth == 18:
            model = ResNet18()
        elif args.depth == 50:
            model = ResNet50()
        else:
            sys.exit("resnet doesn't implement those depth!")
    # elif args.arch == "convnet":
    #     args.depth = 4
    #     model = ConvNet()
    #     print("convnet selected")
    # elif args.arch == "lenet":
    #     args.depth = 5
    #     model = LeNet()
    elif args.arch == "mobilenet":
        args.depth = 13
        model = MobileNet()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    model.cuda()

model = AttackPGD(model)



#############

criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
# args.smooth = args.smooth_eps > 0.0
# args.mixup = config.alpha > 0.0

optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

optimizer = None
if(args.optmzr == 'sgd'):
    optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr,momentum=0.9, weight_decay=1e-4)
elif(args.optmzr =='adam'):
    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)



scheduler = None
if args.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=4e-08)
elif args.lr_scheduler == 'default':
    # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
    epoch_milestones = [80,150]

    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*len(train_loader) for i in epoch_milestones], gamma=0.1)
else:
    raise Exception("unknown lr scheduler")

if args.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr/args.warmup_lr, total_iter=args.warmup_epochs*len(train_loader), after_scheduler=scheduler)


#############

def train(train_loader,criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        scheduler.step()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        # compute output
        nat_output,adv_output,pert_inputs = model(input,target)

        if args.mixup:
            adv_loss = mixup_criterion(criterion, adv_output, target_a, target_b, lam, args.smooth)
            nat_loss = mixup_criterion(criterion, nat_output, target_a, target_b, lam, args.smooth)
        else:
            adv_loss = criterion(adv_output, target, smooth=args.smooth)
            nat_loss = criterion(nat_output, target, smooth=args.smooth)

            # l1_reg_loss = L1Regularizer(model=model, lambda_reg=1e-5)
            # ce_loss = l1_reg_loss.regularized_all_param(reg_loss_function=ce_loss)

        # measure accuracy and record loss
        nat_acc1, _ = accuracy(nat_output, target, topk=(1, 5))
        adv_acc1, _ = accuracy(adv_output, target, topk=(1, 5))

        nat_losses.update(nat_loss.item(), input.size(0))
        adv_losses.update(adv_loss.item(), input.size(0))
        adv_top1.update(adv_acc1[0], input.size(0))
        nat_top1.update(nat_acc1[0], input.size(0))



        # compute gradient and do SGD step
        optimizer.zero_grad()
        adv_loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                  'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                  'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                  'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
            .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, nat_loss=nat_losses, nat_top1=nat_top1, adv_loss=adv_losses, adv_top1=adv_top1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(model, criterion, test_loader):
    model.eval()
    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()

    nat_correct = 0
    adv_correct = 0
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            nat_output,adv_output,pert_inputs = model(data, target)
            nat_loss = criterion(nat_output, target)
            adv_loss = criterion(adv_output, target)

            nat_losses.update(nat_loss.item(), data.size(0))
            adv_losses.update(adv_loss.item(), data.size(0))

            nat_pred = nat_output.max(1, keepdim=True)[1]
            adv_pred = adv_output.max(1, keepdim=True)[1]

            nat_correct += nat_pred.eq(target.view_as(nat_pred)).sum().item()
            adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('\nTest set nat_loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
        nat_losses.avg, nat_correct, len(test_loader.dataset),
        100. * float(nat_correct) / float(len(test_loader.dataset))))
    print('\nTest set adv_loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
        adv_losses.avg, adv_correct, len(test_loader.dataset),
        100. * float(adv_correct) / float(len(test_loader.dataset))))

    nat_acc = 100. * float(nat_correct) / float(len(test_loader.dataset))
    adv_acc = 100. * float(adv_correct) / float(len(test_loader.dataset))

    return (nat_acc+adv_acc)/2, adv_acc

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def main():
    all_nat_acc = [0.000]
    all_adv_acc = [0.000]
    for epoch in range(0, args.epochs):
        # if epoch in [args.epochs * 0.26, args.epochs * 0.4, args.epochs * 0.6, args.epochs * 0.83]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        # if args.lr_scheduler == "default":
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * (0.5 ** (epoch // args.lr_decay))
        # elif args.lr_scheduler == "cosine":
        #     scheduler.step()

        train(train_loader,criterion, optimizer, epoch)
        nat_acc,adv_acc = test(model, criterion, test_loader)

        if nat_acc > max(all_nat_acc):
            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(nat_acc))
            torch.save(model.state_dict(), "./model/cifar10_{}{}_avg_acc_{:.3f}_{}.pt".format(args.arch, args.depth, nat_acc, args.optmzr))
            print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(all_nat_acc)))
            if len(all_nat_acc) > 1:
                os.remove("./model/cifar10_{}{}_avg_acc_{:.3f}_{}.pt".format(args.arch, args.depth, max(all_nat_acc), args.optmzr))
        all_nat_acc.append(nat_acc)

        if adv_acc > max(all_adv_acc):
            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(adv_acc))
            torch.save(model.state_dict(), "./model/cifar10_{}{}_adv_acc_{:.3f}_{}.pt".format(args.arch, args.depth, adv_acc, args.optmzr))
            print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(all_adv_acc)))
            if len(all_adv_acc) > 1:
                os.remove("./model/cifar10_{}{}_adv_acc_{:.3f}_{}.pt".format(args.arch, args.depth, max(all_adv_acc), args.optmzr))
        all_adv_acc.append(adv_acc)

        if adv_acc >= 99.99:
            print("accuracy is high ENOUGH!")
            break

    print("Best avg accuracy: " + str(max(all_nat_acc)))
    print("Best adv accuracy: " + str(max(all_adv_acc)))


if __name__ == '__main__':
    main()