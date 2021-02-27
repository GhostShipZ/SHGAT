import numpy as np
import os
import logging
import pandas as pd

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    statistics = np.load('../data/' + args.dataset + '/statistics.npy',allow_pickle=True).item()
    user_num_old, user_num_new, item_num, entity_num, relation_num, triple_num_old, triple_num_new, train_num, val_num, test_num = statistics['user_num_old'], statistics['user_num_new'], statistics['item_num'], statistics['entity_num'], statistics['relation_num'], statistics['triple_num_old'], statistics['triple_num_new'], statistics['train_num'], statistics['val_num'], statistics['test_num']

    adj_entity, adj_relation = load_kg(args, entity_num, relation_num)
    items_clicked, num_clicked = load_user_clicked(args,user_num_old,entity_num)
    logging.info('data loaded.')
    return entity_num, relation_num, adj_entity, adj_relation, items_clicked, num_clicked, user_num_old, item_num


def load_kg(args, n_entity, n_relation):
    logging.info('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity, n_relation)

    return adj_entity, adj_relation


def construct_kg(kg_np):
    logging.info('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg

def construct_adj(args, kg, entity_num, relation_num):
    logging.info('constructing adjacency matrix ...')
    nei_len = [len(i) for i in kg.values()]
    # Coarse sampling of entity neighbors.
    sample_size = int(np.percentile(nei_len,args.neighbor_percentile))
    adj_entity = np.full((entity_num+1, sample_size), entity_num, dtype=np.int64) # Index entity_num and relation_num represent the empty node and edge
    adj_relation = np.full((entity_num+1, sample_size), relation_num, dtype=np.int64)

    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors > sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=sample_size, replace=False)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
        else:
            adj_entity[entity][:n_neighbors] = np.array([i[0] for i in neighbors])
            adj_relation[entity][:n_neighbors] = np.array([i[1] for i in neighbors])
    return adj_entity, adj_relation


def load_user_clicked(args,n_user,n_entity):
    logging.info('sampling items clicked by users...')
    ratingfile = '../data/' + args.dataset + '/train.npy'
    ratings = np.load(ratingfile)
    ratings = ratings[ratings[:,2]==1,:]

    df = pd.DataFrame(ratings[:,:-1],columns=('user','item'))
    user_item = df.groupby('user')['item'].agg(list).reset_index()
    user_item['items_len'] = user_item['item'].apply(len)
    items_len = user_item['items_len'].tolist()
    # Coarse sampling of interaction sets.
    sample_size = int(np.percentile(items_len,args.user_click_percentile))
    items_clicked = np.full((n_user,sample_size),n_entity,dtype=np.int64)
    num_clicked = np.zeros((n_user,1))
    for user,items in zip(user_item['user'],user_item['item']):
        num = len(items)
        if num > sample_size:
            sampled_indices = np.random.choice(list(range(num)),size=sample_size,replace=False)
            items_clicked[user] = np.array([items[i] for i in sampled_indices])
            num_clicked[user] = sample_size
        else:
            items_clicked[user][:num] = np.array(items)
            num_clicked[user] = num
    return items_clicked, num_clicked