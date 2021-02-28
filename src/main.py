import argparse
import numpy as np
from time import time
from data_loader import load_data
import torch
import os
import random

# def seed(seed=43):
#     seed = int(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = False

# seed(444)

parser = argparse.ArgumentParser()


# movie-20M
parser.add_argument('--dataset', type=str, default='movie-20M', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=128, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=8192, help='batch size') 
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
parser.add_argument('--neighbor_percentile', type=float, default=90, help='coarse sample of entity neighbors')
parser.add_argument('--user_click_percentile', type=float, default=99, help='coarse sampling of interaction sets')
parser.add_argument('--device', type=str, default='0', help='GPU index')


'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1.5e-4, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
parser.add_argument('--neighbor_percentile', type=float, default=95, help='coarse sampling of entity neighbors')
parser.add_argument('--user_click_percentile', type=float, default=99, help='coarse sampling of interaction sets')
parser.add_argument('--device', type=str, default='0', help='GPU index')
'''

show_loss = False
show_topk = True


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
from train import train
data = load_data(args)
train(args, data, show_loss, show_topk)