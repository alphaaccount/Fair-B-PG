import scipy.sparse as sp
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from args import parse_args
from utils import fix_seed, find_link
from dataloader import get_dataset
from model.utils import preprocess_graph, project
from model.optimizer import loss_function
from model.gae import GCNModelVAE
from eval import fair_link_eval
import random
from scipy.sparse import lil_matrix
import json
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')    
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        print('.', end=' ')
        return 0.
    return dcg_at_k(r, k) / idcg

def find_dp_ndcg():
    data = pd.read_csv('allPokec_age_gender_bins2.csv', sep=',')
    #adj = np.genfromtxt("adjacency.txt",skip_header=0,dtype=float)
    adj = np.load('adjacency.npz')
    with open('reverse_map.txt','r') as r:
      r1 = r.read()
    with open('nodelist.txt','r') as n:
      n1 = n.read()
    rev = eval(r1)
    nodelist = eval(n1)
    actual = {}  
    nodelist_rev = {}
    for key in nodelist:
      x = nodelist[key]
      nodelist_rev[x] = key

    for key in rev:
      x = rev[key]
      actual[x] = key
    #print(actual)
    print(actual.keys())
    #print(nodelist.keys())
    m = 4
    n = len(data)
    for i in range(n):
      n1 = int(data.iloc[i,0])
      n2 = int(data.iloc[i,3])
      if n1 in actual.keys() and n2 in actual.keys():
        m1 = actual[n1]
        m2 = actual[n2]
        if m1 not in nodelist_rev.keys() or m2 not in nodelist_rev.keys():
          data = data.drop(i,axis=0)
      else:
        data = data.drop(i, axis=0) 
    print(len(data)) 
    n = len(data)
    data1 = np.zeros((m,n))
    
    graph_embedding = np.genfromtxt("../dataset/pokec/fair_adj.embedding",skip_header=0,dtype=float)
    embedding_df = pd.DataFrame(graph_embedding)
    #embedding_df = embedding_df.rename(index=int, columns={0:"user_id"})
    print(embedding_df)
    #user_ids = embedding_df['user_id']
    #embedding_df = embedding_df.drop(['user_id'],axis=1)
    embedding_np = embedding_df.to_numpy()
    sizes = np.zeros(m)
    fair_scores = []
    
    gr = [ [] for _ in range(m) ]
    grf= [ [] for _ in range(m) ]
    
    #count = np.zeros(r, dtype=int)
    fair_scores = []
    DP_list = []
    DPf_list = []
    
    for i in range(n):
          fair_scores.append(sigmoid(np.inner(embedding_np[nodelist_rev[actual[int(data.iloc[i,0])]]],embedding_np[nodelist_rev[actual[int(data.iloc[i,3])]]])))
          for j in range(2):
            for k in range(2):
              if data.iloc[i,1] == j and data.iloc[i,4] == k:
                gr[2*j+k].append(data.iloc[i,6])
                grf[2*j+k].append(fair_scores[i])
                data1[2*j+k][i] = 1
    #print(fair_scores)            
    max_size = 0
    for i in range(m):
      count=0
      for j in range(n):
        if data1[i][j]==1:
          count=count+1 
      if count>max_size:
        max_size=count
      sizes[i]=count
    print(sizes) 

    for i in range(m):
      DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
      DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

    print(max(DP_list), min(DP_list))
    print(max(DPf_list), min(DPf_list))
    DP = max(DP_list)-min(DP_list)
    DPf = max(DPf_list)-min(DPf_list)
    print("Demogrphic Parity is without/with fair", DP, DPf)

    M = embedding_np.shape[0]
    k = 10
    accum_ndcg = 0
    node_cnt = 0
    accum_ndcg_u = 0
    node_cnt_u = 0    
    adj_mtx_fair = lil_matrix((M,M))
    adj_mtx_unfair = lil_matrix((M,M))
    selected_pairs = lil_matrix((M,M))

    for i in range(n):
        adj_mtx_unfair[nodelist_rev[actual[int(data.iloc[i,0])]],nodelist_rev[actual[int(data.iloc[i,3])]]] = data.iloc[i,6]
        selected_pairs[nodelist_rev[actual[int(data.iloc[i,0])]],nodelist_rev[actual[int(data.iloc[i,3])]]] = 1
        adj_mtx_fair[nodelist_rev[actual[int(data.iloc[i,0])]],nodelist_rev[actual[int(data.iloc[i,3])]]] = fair_scores[i]

    #print(adj_mtx_fair)
    #print(np.count_nonzero(adj_mtx_fair))
    print('Utility evaluation (link prediction)')
    s = random.sample(range(M),4000)
    for node_id in s:
        node_edges = adj[node_id]
        test_pos_nodes = []
        neg_nodes = []
        for i in range(M):
            if selected_pairs[node_id,i]==1:
                 if adj[node_id,i]>0:
                     test_pos_nodes.append(i)
                 else:
                     neg_nodes.append(i)
 
        #pred_edges_fair.append(adj_mtx_fair[node_id][i])
        #pred_edges_unfair.append(adj_mtx_fair[node_id][i])

        #pos_nodes = np.where(node_edges>0)[0]
        #num_pos = len(pos_nodes)
        #num_test_pos = int(len(pos_nodes) / 10) + 1
        #test_pos_nodes = pos_nodes[:num_test_pos]
        num_pos = len(test_pos_nodes)
        all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes)) 
        all_eval_edges = np.zeros(20)  # because each node has 20 neighbors in the set


        all_eval_edges[:num_pos] = 1
        #print(all_eval_edges)


        #in pred_edges all positive edges should be before and then negative edge sores
        edges = []
        pred_edges_fair_pos = []
        pred_edges_fair_neg = []

        pred_edges_unfair_pos = []
        pred_edges_unfair_neg = []

        for i in range(M):
            if int(selected_pairs[node_id,i])==1:
                 if adj[node_id,i]>0:
                     pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                 else:
                     pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])

        #print(pred_edges_fair_pos)
        pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
        pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
        if len(pred_edges_unfair) >=k:
            #pred_edges_unfair = np.array(pred_edges_unfair)
            rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg_u = ndcg_at_k(ranked_node_edges, k)
            if ndcg_u != 0.0:
                 accum_ndcg_u += ndcg_u
                 print(ndcg_u, node_cnt_u)
                 node_cnt_u += 1
 
        if len(pred_edges_fair) >=k:
            #pred_edges_fair = np.array(pred_edges_fair)
            rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg = ndcg_at_k(ranked_node_edges, k)
            if ndcg != 0.0:
                 accum_ndcg += ndcg
                 node_cnt += 1

    score = accum_ndcg/node_cnt
    score_u = accum_ndcg_u/node_cnt_u

    # now compute accuracy as well and dp

    print('-- ndcg of link prediction for Adj score:{}'.format(score))
    print('-- ndcg of link prediction for unfair score:{}'.format(score_u))

find_dp_ndcg()