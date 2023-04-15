from __future__ import print_function
import os
import sys
import logging
import pickle
import yaml
import argparse
import collections
import time
from time import strftime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from vgg import VGG
from mobilenet import MobileNet
from resnet import ResNet18, ResNet50

import prune_util
from prune_util import GradualWarmupScheduler
from prune_util import CrossEntropyLossMaybeSmooth
from prune_util import mixup_data, mixup_criterion

from testers import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 rew training')
parser.add_argument('--logger', action='store_true', default=True,
                    help='whether to use logger')
parser.add_argument('--arch', type=str, default="vgg",
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=True,
                    help='for multi-gpu training')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='sgd', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--masked-retrain', action='store_true', default=False,
                    help='for masked retrain')
parser.add_argument('--verbose', action='store_true', default=True,
                    help='whether to report rew convergence condition')
parser.add_argument('--rew', action='store_true', default=False,
                    help="for reweighted l1 training")
parser.add_argument('--masked-grow', action='store_true', default=False,
                    help="for masked growing")
parser.add_argument('--sparsity-type', type=str, default='irregular',
                    help="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--config-file', type=str, default='config_resnet18v1',
                    help="config file name")
parser.add_argument('--combine-progressive', action='store_true', default=False,
                    help="for filter pruning after column pruning")
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
parser.add_argument('--alpha', type=float, default=0.0, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
parser.add_argument('--no-tricks', action='store_true', default=False,
                    help='disable all training tricks and restore original classic training process')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(args.seed)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
writer = None

if args.logger:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    try:
        os.makedirs("logger", exist_ok=True)
    except TypeError:
        raise Exception("Direction not create!")
    logger.addHandler(logging.FileHandler(strftime('logger/CIFAR_%m-%d-%Y-%H:%M.log'), 'a'))
    global print
    print = logger.info

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# set up model archetecture
if args.arch == "vgg":
    if args.depth == 16:
        model = VGG(depth=16, init_weights=True, cfg=None)
        model_record = VGG(depth=16, init_weights=True, cfg=None)
    elif args.depth == 19:
        model = VGG(depth=19, init_weights=True, cfg=None)
        model_record = VGG(depth=19, init_weights=True, cfg=None)
    else:
        sys.exit("vgg doesn't have those depth!")
elif args.arch == "resnet":
    if args.depth == 18:
        model = ResNet18()
        model_record = ResNet18()
    elif args.depth == 50:
        model = ResNet50()
        model_record = ResNet50()
    else:
        sys.exit("resnet doesn't implement those depth!")
elif args.arch == "mobilenet":
    args.depth = 13
    model = MobileNet()
    model_record = MobileNet()
if args.multi_gpu:
    model = torch.nn.DataParallel(model)
    model_record = torch.nn.DataParallel(model_record)
model.cuda()
model_record.cuda()

""" disable all bag of tricks"""
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = "default"
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0


def main():
    print(args)

    """ bag of tricks set-ups"""
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

    optimizer = None
    if args.optmzr == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    elif args.optmzr == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                         eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [39, 79, 119, 159, 199, 239, 279]

        """Set the learning rate of each parameter group to the initial lr decayed
            by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[i * len(train_loader) for i in epoch_milestones],
                                                   gamma=0.7)
    else:
        raise Exception("unknown lr scheduler")

    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr,
                                           total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)

    """====================="""
    """ reweighted L1 train """
    """====================="""
    if args.rew:
        reweighted_training(criterion, optimizer, scheduler)

    """=============="""
    """masked retrain"""
    """=============="""
    if args.masked_retrain:
        masked_retrain(criterion, optimizer, scheduler)


def convert_adv_input(model, input, target):
    basic_model = model
    rand = True
    step_size = 2.0 / 255
    epsilon = 8.0 / 255
    num_steps = 10

    x = input.detach()
    if rand:
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = basic_model(x)
            loss = F.cross_entropy(logits, target, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, input - epsilon), input + epsilon)
        x = torch.clamp(x, 0, 1)

    return x


def reweighted_training(criterion, optimizer, scheduler):
    original_model_name = "./model/cifar10_resnet18_pretrained.pt"
    print("\n>_ Loading baseline/progressive model..... {}\n".format(original_model_name))
    model.load_state_dict(torch.load(original_model_name))  # need basline model

    model_record_name = "./model/cifar10_resnet18_pretrained.pt"

    model_record.load_state_dict(torch.load(model_record_name))

    layers = []
    for name, weight in model.named_parameters():
        # if 'conv' in name:
        #     layers.append(module)
        # elif 'fc' in name:
        #     layers.append(module)
        if args.arch == "vgg":
            if (len(weight.size()) == 4):
                layers.append(weight)
        elif args.arch == "resnet":
            if (len(weight.size()) == 4):
                layers.append(weight)
        elif args.arch == "mobilenet":
            pass

    eps = 1e-3

    # initialize rew_layer
    rew_layers = []
    for i in range(len(layers)):
        conv_layer = layers[i]
        if args.sparsity_type == "irregular":
            rew_layers.append(1 / (conv_layer.data + eps))
        elif args.sparsity_type == "column":
            rew_layers.append(1 / (torch.norm(conv_layer.data, dim=0) + eps))
        elif args.sparsity_type == "kernel":
            rew_layers.append(1 / (torch.norm(conv_layer.data, dim=[2, 3]) + eps))
        elif args.sparsity_type == "filter":
            rew_layers.append(1 / (torch.norm(torch.norm(conv_layer.data, dim=1), dim=[1, 2]) + eps))

    for epoch in range(1, args.epochs + 1):
        # train(train_loader, criterion, optimizer, scheduler, epoch, args, layers, rew_layers, eps)
        adv_train(train_loader, criterion, optimizer, scheduler, epoch, args, layers, rew_layers, eps)
        t_loss, prec1 = test(model, criterion, test_loader)

        # save_dir = "./model_reweighted/mnist_reweighted_eps_{}_acc_{}.pt".format(eps, prec1)
        save_dir = "./model_reweighted/temp3.pt"
        print("Saving model... {}\n".format(save_dir))
        torch.save(model.state_dict(), save_dir)


def masked_retrain(criterion, optimizer, scheduler):
    print("\n>_ Loading file...")
    # model.load_state_dict(torch.load("./model_reweighted/mnist_reweighted_eps_0.0001_acc_98.79.pt"))
    # model.load_state_dict(torch.load("./model_reweighted/temp3.pt"))

    original_model_name = "./model/cifar10_resnet18_pretrained.pt"
    print("\n>_ Loading baseline/progressive model..... {}\n".format(original_model_name))
    model.load_state_dict(torch.load(original_model_name))

    model.cuda()

    model_record_name = "./model/cifar10_resnet18_pretrained.pt"

    model_record.load_state_dict(torch.load(model_record_name))

    best_prec1 = [0]

    prune_ratios = []
    with open("./profile/" + args.config_file + ".yaml", "r") as stream:
        try:
            raw_dict = yaml.full_load(stream)
            prune_ratios = raw_dict['prune_ratios']
        except yaml.YAMLError as exc:
            print(exc)

    if args.masked_grow:
        reverse_masks = {}
        weight_record = {}
        for name, W in (model_record.named_parameters()):
            if (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                weight = W.cpu().detach().numpy()
                zeros = weight == 0
                zeros = zeros.astype(np.float32)
                non_zero_mask = torch.from_numpy(zeros).cuda()
                W = torch.from_numpy(weight).cuda()
                W.data = W
                reverse_masks[name] = non_zero_mask
                weight_record[name] = W
        for name, W in (model.named_parameters()):
            if name in reverse_masks and (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                W.data *= reverse_masks[name]

    prune_util.hard_prune(args, prune_ratios, model)

    if args.masked_grow:
        for name, W in (model.named_parameters()):
            if name in reverse_masks and (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                W.data += weight_record[name]

    epoch_loss_dict = {}
    testAcc = []

    for epoch in range(1, args.epochs + 1):
        # train(train_loader, criterion, optimizer, scheduler, epoch, args, layers=None, rew_layers=None,
        #       eps=None)
        idx_loss_dict = adv_train(train_loader, criterion, optimizer, scheduler, epoch, args, layers=None,
                                  rew_layers=None,
                                  eps=None)
        t_loss, prec1 = test(model, criterion, test_loader)

        if prec1 > max(best_prec1):
            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            torch.save(model.state_dict(), "./model_retrained2/cifar10_{}{}_retrained_acc_{:.3f}_{}_{}.pt".format(
                args.arch, args.depth, prec1, args.config_file, args.sparsity_type))
            print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_prec1)))
            if len(best_prec1) > 1:
                os.remove("./model_retrained2/cifar10_{}{}_retrained_acc_{:.3f}_{}_{}.pt".format(
                    args.arch, args.depth, max(best_prec1), args.config_file, args.sparsity_type))

        epoch_loss_dict[epoch] = idx_loss_dict
        testAcc.append(prec1)

        best_prec1.append(prec1)
        print("current best acc is: {:.4f}".format(max(best_prec1)))

    # test_column_sparsity(model)
    # test_filter_sparsity(model)

    print("Best Acc: {:.4f}%".format(max(best_prec1)))
    # np.save(strftime("./plotable/%m-%d-%Y-%H:%M_plotable_{}.npy".format(args.sparsity_type)), epoch_loss_dict)
    # np.save(strftime("./plotable/%m-%d-%Y-%H:%M_testAcc_{}.npy".format(args.sparsity_type)), testAcc)


re = 1e-4


def adv_train(train_loader, criterion, optimizer, scheduler, epoch, args, layers, rew_layers, eps):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    idx_loss_dict = {}
    adv_losses = AverageMeter()
    adv_top1 = AverageMeter()

    # switch to train mode
    model.train()

    if args.masked_retrain and not args.combine_progressive:
        print("full acc re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask
    elif args.combine_progressive:
        print("progressive rew-train/re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask

    elif args.masked_grow and args.rew:
        print("masked reweighted growing")
        masks = {}
        for name, W in (model.named_parameters()):
            if (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                weight = W.cpu().detach().numpy()
                zeros = weight == 0
                zeros = zeros.astype(np.float32)
                non_zero_mask = torch.from_numpy(zeros).cuda()
                W = torch.from_numpy(weight).cuda()
                W.data = W
                masks[name] = non_zero_mask

    if args.masked_retrain and args.masked_grow:
        reverse_masks = {}
        weight_record = {}
        for name, W in (model_record.named_parameters()):
            if (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                weight = W.cpu().detach().numpy()
                zeros = weight == 0
                zeros = zeros.astype(np.float32)
                non_zero_mask = torch.from_numpy(zeros).cuda()
                W = torch.from_numpy(weight).cuda()
                W.data = W
                reverse_masks[name] = non_zero_mask
                weight_record[name] = W

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

        adv_input = convert_adv_input(model, input, target)

        # compute output
        output = model(input)
        adv_output = model(adv_input)

        if args.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
            adv_loss = mixup_criterion(criterion, adv_output, target_a, target_b, lam, args.smooth)
        else:
            ce_loss = criterion(output, target, smooth=args.smooth)
            adv_loss = criterion(adv_output, target, smooth=args.smooth)

        if args.rew:
            if i == 0:
                print("reweighted l1 training...\n")
            rew_milestone = [40, 80, 120, 160]
            # adjust_rew_learning_rate2(optimizer, epoch, rew_milestone, args)

            l1_loss = 0
            # add reweighted l1 loss
            if i == 0 and epoch in rew_milestone:
                print("reweighted l1 update")
                for j in range(len(layers)):
                    if args.sparsity_type == "irregular":
                        rew_layers[j] = (1 / (layers[j].data + eps))
                    elif args.sparsity_type == "column":
                        rew_layers[j] = (1 / (torch.norm(layers[j].data, dim=0) + eps))
                    elif args.sparsity_type == "kernel":
                        rew_layers[j] = (1 / (torch.norm(layers[j].data, dim=[2, 3]) + eps))
                    elif args.sparsity_type == "filter":
                        rew_layers[j] = (1 / (torch.norm(torch.norm(layers[j].data, dim=1), dim=[1, 2]) + eps))

            for j in range(len(layers)):
                rew = rew_layers[j]
                conv_layer = layers[j]
                if args.sparsity_type == "irregular":
                    l1_loss = l1_loss + 1e-8 * torch.sum((torch.abs(rew * conv_layer)))
                elif args.sparsity_type == "column":
                    l1_loss = l1_loss + re[j] * torch.sum(rew * torch.norm(conv_layer, dim=0))
                elif args.sparsity_type == "kernel":
                    l1_loss = l1_loss + 1e-5 * torch.sum(rew * torch.norm(conv_layer, dim=[2, 3]))
                elif args.sparsity_type == "filter":
                    l1_loss = l1_loss + 3e-3 * torch.sum(rew * torch.norm(torch.norm(conv_layer, dim=1), dim=[1, 2]))
            ce_loss = l1_loss + ce_loss

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        adv_acc1, _ = accuracy(adv_output, target, topk=(1, 5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        adv_losses.update(adv_loss.item(), input.size(0))
        adv_top1.update(adv_acc1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        adv_loss.backward()

        if args.combine_progressive:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]
        if args.masked_retrain and not args.masked_grow:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    # print (name)
                    if name in masks:
                        W.grad *= masks[name]
        if args.masked_grow and args.rew:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]

        if args.masked_grow and args.masked_retrain:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                        W.grad *= (masks[name] * reverse_masks[name])

        optimizer.step()

        if args.masked_grow and args.masked_retrain:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                        W.data = W.data * reverse_masks[name] + weight_record[name]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print(i)
        if i % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print('({0}) lr:[{1:.5f}]  '
                  'Epoch: [{2}][{3}/{4}]\t'
                  'Status: rew-[{5}] retrain-[{6}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                  'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                  .format(args.optmzr, current_lr,
                          epoch, i, len(train_loader), args.rew, args.masked_retrain, batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, adv_loss=adv_losses, adv_top1=adv_top1))

        if i % 100 == 0:
            idx_loss_dict[i] = adv_losses.avg
    return idx_loss_dict


def train(train_loader, criterion, optimizer, scheduler, epoch, args, layers, rew_layers, eps):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    idx_loss_dict = {}

    # switch to train mode
    model.train()

    if args.masked_retrain and not args.combine_progressive:
        print("full acc re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask
    elif args.combine_progressive:
        print("progressive rew-train/re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask

    elif args.masked_grow and args.rew:
        print("masked reweighted growing")
        masks = {}
        for name, W in (model.named_parameters()):
            if (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                weight = W.cpu().detach().numpy()
                zeros = weight == 0
                zeros = zeros.astype(np.float32)
                non_zero_mask = torch.from_numpy(zeros).cuda()
                W = torch.from_numpy(weight).cuda()
                W.data = W
                masks[name] = non_zero_mask

    if args.masked_retrain and args.masked_grow:
        reverse_masks = {}
        weight_record = {}
        for name, W in (model_record.named_parameters()):
            if (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                weight = W.cpu().detach().numpy()
                zeros = weight == 0
                zeros = zeros.astype(np.float32)
                non_zero_mask = torch.from_numpy(zeros).cuda()
                W = torch.from_numpy(weight).cuda()
                W.data = W
                reverse_masks[name] = non_zero_mask
                weight_record[name] = W

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
        output = model(input)

        if args.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
        else:
            ce_loss = criterion(output, target, smooth=args.smooth)

        if args.rew:
            if i == 0:
                print("reweighted l1 training...\n")
            rew_milestone = [40, 80, 120, 160]
            # adjust_rew_learning_rate2(optimizer, epoch, rew_milestone, args)

            l1_loss = 0
            # add reweighted l1 loss
            if i == 0 and epoch in rew_milestone:
                print("reweighted l1 update")
                for j in range(len(layers)):
                    if args.sparsity_type == "irregular":
                        rew_layers[j] = (1 / (layers[j].data + eps))
                    elif args.sparsity_type == "column":
                        rew_layers[j] = (1 / (torch.norm(layers[j].data, dim=0) + eps))
                    elif args.sparsity_type == "kernel":
                        rew_layers[j] = (1 / (torch.norm(layers[j].data, dim=[2, 3]) + eps))
                    elif args.sparsity_type == "filter":
                        rew_layers[j] = (1 / (torch.norm(torch.norm(layers[j].data, dim=1), dim=[1, 2]) + eps))

            for j in range(len(layers)):
                rew = rew_layers[j]
                conv_layer = layers[j]
                if args.sparsity_type == "irregular":
                    l1_loss = l1_loss + 1e-8 * torch.sum((torch.abs(rew * conv_layer)))
                elif args.sparsity_type == "column":
                    l1_loss = l1_loss + re[j] * torch.sum(rew * torch.norm(conv_layer, dim=0))
                elif args.sparsity_type == "kernel":
                    l1_loss = l1_loss + 1e-5 * torch.sum(rew * torch.norm(conv_layer, dim=[2, 3]))
                elif args.sparsity_type == "filter":
                    l1_loss = l1_loss + 3e-3 * torch.sum(rew * torch.norm(torch.norm(conv_layer, dim=1), dim=[1, 2]))
            ce_loss = l1_loss + ce_loss

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # ce_loss.backward()

        if args.combine_progressive:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]
        if args.masked_retrain and not args.masked_grow:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    # print (name)
                    if name in masks:
                        W.grad *= masks[name]
        if args.masked_grow and args.rew:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]

        if args.masked_grow and args.masked_retrain:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                        W.grad *= (masks[name] * reverse_masks[name])

        optimizer.step()

        if args.masked_grow and args.masked_retrain:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and (len(W.size()) == 4) and 'layer' in name and 'shortcut' not in name:
                        W.data = W.data * reverse_masks[name] + weight_record[name]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print(i)
        if i % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print('({0}) lr:[{1:.5f}]  '
                  'Epoch: [{2}][{3}/{4}]\t'
                  'Status: rew-[{5}] retrain-[{6}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  .format(args.optmzr, current_lr,
                          epoch, i, len(train_loader), args.rew, args.masked_retrain, batch_time=data_time, loss=losses,
                          top1=top1))
        if i % 100 == 0:
            idx_loss_dict[i] = losses.avg
    return idx_loss_dict


def test(model, criterion, test_loader):
    batch_time = AverageMeter()
    model.eval()
    losses = AverageMeter()
    correct = 0

    adv_losses = AverageMeter()
    adv_correct = 0

    top1 = AverageMeter()
    adv_top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            adv_data = convert_adv_input(model, data, target)
            output = model(data)
            adv_output = model(adv_data)

            loss = criterion(output, target)
            adv_loss = criterion(adv_output, target)

            losses.update(loss.item(), data.size(0))
            adv_losses.update(adv_loss.item(), adv_data.size(0))

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))

            top1.update(acc1[0], data.size(0))
            adv_top1.update(adv_acc1[0], adv_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                      'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                .format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1, adv_loss=adv_losses, adv_top1=adv_top1))

        print(' * Acc@1 {top1.avg:.3f} *Adv_Acc@1 {adv_top1.avg:.3f}'
            .format(top1=top1, adv_top1=adv_top1))

        # global best_mean_loss
        # mean_loss = (adv_losses.avg + losses.avg) / 2
        # if mean_loss < best_mean_loss:
        #     best_mean_loss = mean_loss
        #     print('new best_mean_loss is {}'.format(mean_loss))
    return adv_top1.avg, (100. * float(adv_correct) / float(len(test_loader.dataset)))


            # loss = criterion(output, target)
            # adv_loss = criterion(adv_output, target)
            # losses.update(loss.item(), data.size(0))
            # adv_losses.update(adv_loss.item(), adv_data.size(0))
            # pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # adv_pred = adv_output.max(1, keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()

    # print('\nTest set loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
    #     losses.avg, correct, len(test_loader.dataset),
    #     100. * float(correct) / float(len(test_loader.dataset))))
    #
    # print('\nTest set adv loss: {:.4f},  * Adv Acc@1: {}/{} ({:.2f}%)\n'.format(
    #     adv_losses.avg, adv_correct, len(test_loader.dataset),
    #     100. * float(adv_correct) / float(len(test_loader.dataset))))
    # return adv_losses.avg, (100. * float(adv_correct) / float(len(test_loader.dataset)))


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


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def adjust_rew_learning_rate(optimizer, epoch, rew_milestone, args):
    """
        during each rew cycle which is defined by rew milestone in training function, about every 1/3 of this cycle,
        the lr decrease. On every start of each cycle, the lr will reset to orginal lr.
    """
    lr = None
    if epoch in rew_milestone:
        lr = args.lr
    else:
        for i, item in enumerate(rew_milestone):
            if item > epoch:
                if i == 0:
                    update_epoch = item
                else:
                    update_epoch = item - rew_milestone[i - 1]
                rew_epoch_offset = (epoch - item) % update_epoch
                rew_step = update_epoch / 2  # roughly every 1/3 in one rew cycle.
                lr = args.lr * (0.1 ** (rew_epoch_offset // rew_step))
                break
        if epoch > max(rew_milestone):
            rew_epoch_offset = (epoch - max(rew_milestone)) % (args.epochs - max(rew_milestone))
            rew_step = (args.epochs - max(rew_milestone)) / 2  # roughly every 1/3 in one rew cycle.
            lr = args.lr * (0.1 ** (rew_epoch_offset // rew_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_rew_learning_rate2(optimizer, epoch, rew_milestone, args):
    current_lr = None
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    if epoch in rew_milestone:
        current_lr *= 0.5
    else:
        current_lr *= 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = time.time() - start_time
    need_hour, need_mins, need_secs = convert_secs2time(duration)
    print('total runtime: {:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs))