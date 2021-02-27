import torch
import torch.nn as nn 
import numpy as np
from time import time
from model import SHGAT
import os

from sklearn.metrics import f1_score, roc_auc_score
import logging
from torch.utils.data import DataLoader,TensorDataset
from collections import defaultdict
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for i, gpu in enumerate(gpus):
    tf.config.experimental.set_memory_growth(gpu,True)


def train(args, data, show_loss, show_topk):
    DEVICE_ID = 0
    torch.cuda.set_device(DEVICE_ID)
    device = torch.device("cuda:{}".format(DEVICE_ID) if torch.cuda.is_available() else "cpu")
    train_data, eval_data, test_data = torch.LongTensor(np.load('../data/' + args.dataset + '/train.npy')), torch.LongTensor(np.load('../data/' + args.dataset + '/valid.npy')), torch.LongTensor(np.load('../data/' + args.dataset + '/test.npy'))
    train_dataset = TensorDataset(train_data[:,0],train_data[:,1],train_data[:,2].float())
    eval_dataset = TensorDataset(eval_data[:,0],eval_data[:,1],eval_data[:,2].float())
    test_dataset = TensorDataset(test_data[:,0],test_data[:,1],test_data[:,2].float())
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
    )   

    interaction_table, offset = get_interaction_table(train_data, data[0])
    model, optimizer, loss_func = _init_model(args, data, device, interaction_table, offset)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data.numpy(), test_data.numpy(), data[7])
    
    for step in range(args.n_epochs):
        for user_indices, item_indices, labels in train_loader:
            scores, predicted_labels = model(user_indices.to(device),item_indices.to(device))
            base_loss = loss_func(scores, labels.to(device))
            ls_loss = loss_func(predicted_labels, labels.to(device))
            loss = base_loss + args.ls_weight * ls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if show_loss:
                print("batch loss: ", loss)
        # CTR evaluation
        eval_auc, eval_f1 = ctr_eval(args, model, eval_loader, device)
        test_auc, test_f1 = ctr_eval(args, model, test_loader, device)
        ctr_info = 'epoch %.2d    eval auc: %.4f f1: %.4f    test auc: %.4f f1: %.4f '
        logging.info(ctr_info, step, eval_auc, eval_f1, test_auc, test_f1)

        if show_topk:
            precision, recall = topk_eval(model, user_list, train_record, test_record, item_set, k_list, args.batch_size, device)
            print('precision: ', end='')
            for i in precision:
                print('%.4f\t' % i, end='')
            print()
            print('recall: ', end='')
            for i in recall:
                print('%.4f\t' % i, end='')
            print('\n')


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.numpy().astype(np.int64)
    values = train_data[:, 2].numpy().astype(np.float32)

    # The lookup table is implemented with tf because there is no alternative function in pytorch
    interaction_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
    return interaction_table, offset


def topk_eval(model, user_list, train_record, test_record, item_set, k_list, batch_size, device):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    model.eval()
    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items = torch.LongTensor(test_item_list[start:start + batch_size]).to(device)
            scores, _ = model(torch.LongTensor([user] * batch_size).to(device), items)
            items,scores = items.cpu().numpy(),scores.detach().cpu().numpy()
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            bs = len(test_item_list)-start
            items = torch.LongTensor(test_item_list[start:]).to(device)
            scores, _ = model(torch.LongTensor([user] * bs).to(device), items)
            items,scores = items.cpu().numpy(),scores.detach().cpu().numpy()
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    model.train()
    return precision, recall


def ctr_eval(args, model, loader, device):
    auc_list = []
    f1_list = []
    model.eval()
    for user_indicies, item_indicies, labels in loader:
        scores, predicted_labels = model(user_indicies.to(device),item_indicies.to(device))
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
    model.train()
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def _init_model(args, data, device, interaction_table, offset):
    n_entity, n_relation, adj_entity, adj_relation, items_clicked, num_clicked,n_user = data[0], data[1], data[2], data[3], data[4], data[5], data[6]
    adj_entity = torch.LongTensor(adj_entity).to(device)
    adj_relation = torch.LongTensor(adj_relation).to(device)
    items_clicked = torch.LongTensor(items_clicked).to(device)
    num_clicked = torch.FloatTensor(num_clicked).to(device)
    if args.dataset == 'movie-20M':
        entity_init_emb = torch.FloatTensor(np.load('../data/movie-20M/item_embed_size{}_window30_iter10_notime.npy'.format(args.dim)))
    elif args.dataset == 'music':
        entity_init_emb = torch.FloatTensor(np.load('../data/music/item_embed_size{}_window30_iter10_notime.npy'.format(args.dim)))
    model = SHGAT(args, n_user, n_entity, n_relation, adj_entity, adj_relation, device, entity_init_emb, items_clicked, num_clicked, interaction_table, offset).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    loss_func = nn.BCELoss()
    return model, optimizer, loss_func


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict