import argparse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

np.random.seed(43)

RATING_FILE_NAME = dict({'movie-20M': 'ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie-20M': ',', 'music': '\t'})
THRESHOLD = dict({'movie-20M': 4, 'music': 0})
word2vecsize = dict({'movie-20M': 128, 'music': 16})



def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1



def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        # timestamp = int(array[3]) # ordered sequences can be used to pre-train item embeddings if timestamps are available in a dataset

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            # user_pos_ratings[user_index_old].add((item_index,timestamp))
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    # writer = open('../data/' + DATASET + '/ratings_final_with_ts.txt', 'w', encoding='utf-8')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]
        # pos_set_wo_time =set ()
        
        # for item,timestamp in pos_item_set:
        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
            # writer.write('%d\t%d\t1\t%d\n' % (user_index, item, timestamp))
            # pos_set_wo_time.add(item)
        # unwatched_set = item_set - pos_set_wo_time

        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            # writer.write('%d\t%d\t0\t0\n' % (user_index, item))
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()

    print('splitting dataset ...')
    # rating_file = '../data/' + DATASET + '/ratings_final_with_ts.txt'
    # train_flie = '../data/' + DATASET + '/train_with_ts.npy'
    # valid_file = '../data/' + DATASET + '/valid_with_ts.npy'
    # test_flie = '../data/' + DATASET + '/test_with_ts.npy'
    rating_file = '../data/' + DATASET + '/ratings_final.txt'
    train_flie = '../data/' + DATASET + '/train.npy'
    valid_file = '../data/' + DATASET + '/valid.npy'
    test_flie = '../data/' + DATASET + '/test.npy'
    rating_np = np.loadtxt(rating_file, dtype=np.int64)
    statistics['triple_num_old'] = rating_np.shape[0]
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if RATIO < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    # Delete invalid samples due to dataset segmentation, the proportion is small enough to ignore the impact on the results.
    valid_user = list(set(train_data[train_data[:,2]==1][:,0]))
    train_data = train_data[np.in1d(train_data[:,0],valid_user)]
    eval_data = eval_data[np.in1d(eval_data[:,0],valid_user)]
    test_data = test_data[np.in1d(test_data[:,0],valid_user)]

    statistics['train_num'] = train_data.shape[0]
    statistics['val_num'] = eval_data.shape[0]
    statistics['test_num'] = test_data.shape[0]
    statistics['triple_num_new'] = statistics['train_num'] + statistics['val_num'] + statistics['test_num']
    np.save(train_flie, train_data)
    np.save(valid_file, eval_data)
    np.save(test_flie, test_data)
    statistics['user_num_old'] = user_cnt
    statistics['user_num_new'] = len(valid_user)
    statistics['item_num'] = len(item_set)
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))
    

def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0
    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    for line in open('../data/' + DATASET + '/kg.txt', encoding='utf-8'):
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))
    writer.close()
    statistics['entity_num'] = entity_cnt 
    statistics['relation_num'] = relation_cnt
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


def get_init_item_embeddings(entity_cnt, size, window = 30, iter = 10):
    # rating_file_train = '../data/' + DATASET + '/train_with_ts.npy'
    rating_file_train = '../data/' + DATASET + '/train.npy'
    item_embed_file = '../data/' + DATASET + '/item_embed'
    ratings = np.load(rating_file_train)
    ratings = ratings[ratings[:,2]==1,:]
    df = pd.DataFrame(ratings[:,:-1],columns=('user','item'))
    # df = pd.DataFrame(ratings[:,[0,1,3]],columns=('user','item','time'))
    # df.sort_values('time', inplace=True)
    df['item'] = df['item'].astype(str)
    user_item = df.groupby('user')['item'].agg(list).reset_index()
    print("Training initial embedding of users and items with word2vec...")
    model = Word2Vec(user_item['item'].values, size=size, window=window, min_count=0, workers=30, seed=1997, iter=iter, sg=1, hs=1)
    print("Constructing initials embeddings...")
    item_embedding = np.random.randn(entity_cnt+1, size)
    for word in model.wv.index2word:
        item_embedding[int(word)] = model.wv[word]
    # np.save(item_embed_file+'_size{}_window{}_iter{}_withtime.npy'.format(size, window, iter), item_embedding)
    np.save(item_embed_file+'_size{}_window{}_iter{}_notime.npy'.format(size, window, iter), item_embedding)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='music', help='which dataset to preprocess')
    parser.add_argument('-ratio', type=float, default=1, help='size of training dataset')
    args = parser.parse_args()
    DATASET = args.d
    RATIO = args.ratio
    statistics = {}
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    entity_cnt = convert_kg()

    get_init_item_embeddings(statistics['entity_num'], word2vecsize[DATASET], window=30, iter=10)
    np.save('../data/' + DATASET + '/statistics.npy',statistics)
    print('done')
