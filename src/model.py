import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
import tensorflow as tf



class SHGAT(nn.Module):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation, device, entity_init_emb, items_clicked, num_clicked, interaction_table, offset):
        super(SHGAT,self).__init__()
        self._parse_args(args, adj_entity, adj_relation, device, interaction_table, offset)
        self.n_relation = n_relation
        self.items_clicked = items_clicked
        self.num_clicked = num_clicked
        '''
        We modified the __ init__() function of nn.Embedding, so that the padding still works when using pre-trained weights:

        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)
        '''
        self.entity_emb_matrix = nn.Embedding(n_entity+1, self.dim , padding_idx = -1, _weight = entity_init_emb)
        self.W_relation = nn.ModuleList()
        for _ in range(self.n_relation):
            self.W_relation.append(nn.Linear(self.dim,self.dim,bias=False))
        
    def forward(
        self,
        user_indices: torch.LongTensor,
        item_indices: torch.LongTensor,
    ):  
        # users are represented by averaging the items they have interacted with
        bs = user_indices.shape[0]
        items_for_user_embedding = self.entity_emb_matrix(self.items_clicked[user_indices]) #bs*M*32
        items_num_inv = torch.reciprocal(self.num_clicked[user_indices]) #bs*1
        user_embeddings = torch.sum(items_for_user_embedding,dim=1).mul(items_num_inv).view(bs,1,1,self.dim)

        seeds = item_indices.view(len(item_indices),1,1) #bs*1*1
        entities = [seeds]
        item_embeddings = self.entity_emb_matrix(seeds)
        entities_emb = [item_embeddings] #bs*1*1*32
        relations_emb = []
        scores_us = []
        for i in range(self.n_iter):
            neighbor_entities = self.adj_entity[entities[i]].view([bs, -1, self.n_neighbor_old]) #bs*1*120 | bs*K*120 | bs*K^2*120
            neighbor_relations = self.adj_relation[entities[i]].view([bs, -1, self.n_neighbor_old]) #bs*1*120 | bs*K*120 | bs*K^2*120
            neighbor_entities_embedding = self.entity_emb_matrix(neighbor_entities) #bs*1*120*32 | bs*K*120*32 | bs*K^2*120*32
            
            for r_type in range(self.n_relation):
                idx = (neighbor_relations==r_type)
                if idx.sum()==0:continue
                neighbor_entities_embedding[idx] = torch.tanh(self.W_relation[r_type](neighbor_entities_embedding[idx]))

            scores_u = torch.mean(user_embeddings*neighbor_entities_embedding,dim = -1) #bs*1*120
            rep = (-9e15 * torch.ones_like(scores_u)).to(self.device)
            scores_u = torch.where(scores_u == 0, rep, scores_u)
            scores_u = F.softmax(scores_u, dim = -1)
            scores_u, ids = torch.sort(scores_u, dim = -1, descending = True)
            scores_u, ids = scores_u[:,:,:self.n_neighbor], ids[:,:,:self.n_neighbor]
            rep = (-9e15 * torch.ones_like(scores_u)).to(self.device)
            scores_u = torch.where(scores_u == 0, rep, scores_u)
            scores_u = F.softmax(scores_u, dim = -1)
            scores_us.append(scores_u)
            
            neighbor_entities = torch.gather(neighbor_entities, dim = -1, index = ids) #bs*1*K | bs*K*K | bs*K^2*K
            neighbor_relations = torch.gather(neighbor_relations, dim = -1, index = ids) #bs*1*K | bs*K*K | bs*K^2*K
            neighbor_entities_embedding = self.entity_emb_matrix(neighbor_entities) #bs*1*K*32 | bs*K*K*32 | bs*K^2*K*32

            for r_type in range(self.n_relation):
                idx = (neighbor_relations==r_type)
                if idx.sum()==0:continue
                neighbor_entities_embedding[idx] = torch.tanh(self.W_relation[r_type](neighbor_entities_embedding[idx]))

            entities.append(neighbor_entities) #[bs*1*1 | bs*1*K | bs*K*K | bs*K^2*K]
            entities_emb.append(neighbor_entities_embedding) #[bs*1*1*32 | bs*1*K*32 | bs*K*K*32 | bs*K^2*K*32]

        # label smoothness
        entity_labels = []
        reset_masks = []  # True means the label of this item is reset to initial value during label propagation
        holdout_item_for_user = None

        for entities_per_iter in entities:
            entities_per_iter = tf.reshape(tf.constant(entities_per_iter.cpu().numpy()),[bs,-1])
            users = tf.expand_dims(tf.constant(user_indices.cpu().numpy()), 1)
            user_entity_concat = users * self.offset + entities_per_iter

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            initial_label = self.interaction_table.lookup(user_entity_concat)
            holdout_mask = tf.cast(holdout_item_for_user - user_entity_concat, tf.bool)  # False if the item is held out
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)  # True if the entity is a labeled item
            reset_mask = tf.logical_and(reset_mask, holdout_mask)  # remove held-out items
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)  # label initialization

            reset_mask = torch.tensor(reset_mask.numpy()).to(self.device)
            initial_label = torch.tensor(initial_label.numpy()).to(self.device)
            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        reset_masks = reset_masks[:-1]  # we do not need the reset_mask for the last iteration


        # knowledge propagation
        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                neighbors_agg = torch.mean(scores_us[hop].unsqueeze(-1) * entities_emb[hop+1], dim = 2, keepdim = True).view(entities_emb[hop].shape) #bs*1*1*32 | bs*1*K*32 | bs*K*K*32
                labels_agg = torch.sum(scores_us[hop] * entity_labels[hop+1].view(bs,-1,self.n_neighbor), dim=-1)
                labels_output = reset_masks[hop].float() * entity_labels[hop] + (~reset_masks[hop]).float() * labels_agg
                if i == self.n_iter-1:
                    output = torch.tanh(neighbors_agg + entities_emb[hop])
                else:
                    output = F.relu(neighbors_agg + entities_emb[hop])
                entity_vectors_next_iter.append(output)
                entity_labels_next_iter.append(labels_output)
            entities_emb = entity_vectors_next_iter
            entity_labels = entity_labels_next_iter

        predicted_labels = entity_labels[0].squeeze()
        item_embeddings = entities_emb[0].squeeze().reshape(bs,self.dim)
        user_embeddings = user_embeddings.squeeze().reshape(bs,self.dim)
        scores = (user_embeddings * item_embeddings).sum(dim=1)
        scores = torch.sigmoid(scores)
        predicted_labels = torch.sigmoid(predicted_labels-0.5)
        return scores, predicted_labels


    def _parse_args(self, args, adj_entity, adj_relation, device, interaction_table, offset):
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.device = device
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.n_neighbor_old = self.adj_entity.shape[1]
        self.interaction_table = interaction_table
        self.offset = offset