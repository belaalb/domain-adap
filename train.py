'''
https://github.com/fungtion/DANN
'''

from __future__ import print_function

import argparse
import os
import sys
import random

import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import models as models_
from train_loop import TrainLoop
from data_loader import Loader

parser = argparse.ArgumentParser(description='Reproducing DANN')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--source', choices=['mnist', 'mnist_m'], default='mnist', help='Path to source data')
parser.add_argument('--target', choices=['mnist', 'mnist_m'], default='mnist_m', help='Path to target data')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

# Setting seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

source_path = os.path.join('.', 'data', args.source)
target_path = os.path.join('.', 'data', args.target)

image_size = 28
img_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

img_transform_mnist = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
source_dataset = datasets.MNIST(root=source_path, download=True, train=True, transform=img_transform_mnist)
source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

train_list = os.path.join(target_path, 'mnist_m_train_labels.txt')
target_dataset = Loader(data_root=os.path.join(target_path, 'mnist_m_train'), data_list=train_list, transform=img_transform)
target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

model = models_.CNNModel()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.cuda:
	model = model.cuda()
	torch.backends.cudnn.benchmark=True

trainer = TrainLoop(model, optimizer, source_loader, target_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, target_name = args.target)

print('Cuda Mode: {}'.format(args.cuda))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('Source: {}'.format(args.source))
print('Target: {}'.format(args.target))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)


