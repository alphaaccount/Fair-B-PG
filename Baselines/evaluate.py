from re import findall
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle as pk
import argparse
import pandas as pd
import os
import json
import random

from utils import dcg_at_k, ndcg_at_k
from data_loader import SENSITIVE_ATTR_DICT  # predefined sensitive attributes for different datasets
from data_loader import DATA_FOLDER, RAW_FOLDER
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# load predefined sensitive attributes for pokec data
# csv format for each line: node_idx, gender, region, age

def load_node_attributes_twitter(attribute_file):
    node_attributes = {}
    opinion_group = {}

    with open(attribute_file, 'r') as fin:
        lines = fin.readlines()
        idx = 0
        for line in lines:
            if idx == 0:
                idx += 1
                continue
            eachline = line.strip().split(',')
            node_idx = int(idx-1)
            opinion = int(float(eachline[0]))
            #region = int(eachline[1])
            #age = int(float(eachline[2]))
            node_attributes[node_idx] = opinion
            
            if opinion not in opinion_group:
                opinion_group[opinion] = []
            opinion_group[opinion].append(node_idx)

            idx += 1

    return node_attributes, {'opinion':opinion_group}


def load_node_attributes_dblp(attribute_file):
    node_attributes = {}
    continent_group = {}

    with open(attribute_file, 'r') as fin:
        lines = fin.readlines()
        idx = 0
        for line in lines:
            if idx == 0:
                idx += 1
                continue
            eachline = line.strip().split(',')
            node_idx = int(idx-1)
            continent = int(float(eachline[0]))
            #region = int(eachline[1])
            #age = int(float(eachline[2]))
            node_attributes[node_idx] = opinion

            if continent not in continent_group:
                continent_group[continent] = []
            continent_group[continent].append(node_idx)

            idx += 1

    return node_attributes, {'continent':continent_group}




def load_node_attributes_polblog(attribute_file):
    node_attributes = {}
    party_group = {}

    with open(attribute_file, 'r') as fin:
        lines = fin.readlines()
        idx = 0
        for line in lines:
            if idx == 0:
                idx += 1
                continue
            eachline = line.strip().split(',')
            node_idx = int(idx-1)
            party = int(float(eachline[0]))
            #region = int(eachline[1])
            #age = int(float(eachline[2]))
            node_attributes[node_idx] = party

            if party not in party_group:
                party_group[party] = []
            party_group[party].append(node_idx)

            idx += 1

    return node_attributes, {'party':party_group}

def load_node_attributes_cora(attribute_file):
    node_attributes = {}
    topic_group = {}

    with open(attribute_file, 'r') as fin:
        lines = fin.readlines()
        idx = 0
        for line in lines:
            if idx == 0:
                idx += 1
                continue
            eachline = line.strip().split(',')
            node_idx = int(idx-1)
            topic = int(float(eachline[0]))
            #region = int(eachline[1])
            #age = int(float(eachline[2]))
            node_attributes[node_idx] = topic

            if topic not in topic_group:
                topic_group[topic] = []
            topic_group[topic].append(node_idx)

            idx += 1

    return node_attributes, {'topic':topic_group}



def load_adjacency_matrix_dblp(file, M):
    adj_mtx = lil_matrix((M,M))
    # load the adjacency matrix of size MxM of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        idx = 0 
        for line in lines:
            if idx == 0: 
                idx += 1
                continue
            eachline = line.strip().split(',')
            scr_node = int(eachline[0])
            dst_node = int(eachline[1])
            weight = float(eachline[2])
            adj_mtx[scr_node, dst_node] = weight 
            adj_mtx[dst_node, scr_node] = weight 
            idx += 1
    
    return adj_mtx


def load_adjacency_matrix_twitter(file, M):
    adj_mtx = np.zeros((M,M))
    # load the adjacency matrix of size MxM of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        idx = 0 
        for line in lines:
            if idx == 0: 
                idx += 1
                continue
            eachline = line.strip().split(',')
            scr_node = int(eachline[0])
            dst_node = int(eachline[1])
            weight = float(eachline[2])
            adj_mtx[scr_node, dst_node] = weight 
            adj_mtx[dst_node, scr_node] = weight 
            idx += 1
    
    return adj_mtx

def load_adjacency_matrix_polblog(file, M):
    adj_mtx = np.zeros((M,M))
    # load the adjacency matrix of size MxM of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        idx = 0 
        for line in lines:
            if idx == 0: 
                idx += 1
                continue
            eachline = line.strip().split(',')
            scr_node = int(eachline[0])
            dst_node = int(eachline[1])
            weight = float(eachline[2])
            adj_mtx[scr_node, dst_node] = weight 
            adj_mtx[dst_node, scr_node] = weight 
            idx += 1
    
    return adj_mtx

def load_adjacency_matrix_nba(file, M):
    adj_mtx = np.zeros((M,M))
    # load the adjacency matrix of size MxM of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        idx = 0 
        for line in lines:
            if idx == 0: 
                idx += 1
                continue
            eachline = line.strip().split(',')
            scr_node = int(eachline[0])
            dst_node = int(eachline[1])
            weight = float(eachline[2])
            adj_mtx[scr_node, dst_node] = weight 
            adj_mtx[dst_node, scr_node] = weight 
            idx += 1
    
    return adj_mtx


def load_node_attributes_pokec(attribute_file):
    node_attributes = {}
    gender_group = {}
    region_group = {}
    age_group = {}

    with open(attribute_file, 'r') as fin:
        lines = fin.readlines()
        idx = 0
        for line in lines:
            if idx == 0:
                idx += 1
                continue
            eachline = line.strip().split(',')
            node_idx = int(idx-1)
            gender = int(float(eachline[0]))
            region = int(eachline[1])
            age = int(float(eachline[2]))
            node_attributes[node_idx] = (gender, region, age)
            
            if gender not in gender_group:
                gender_group[gender] = []
            gender_group[gender].append(node_idx)

            if region not in region_group:
                region_group[region] = []
            region_group[region].append(node_idx)

            if age not in age_group:
                age_group[age] = []
            age_group[age].append(node_idx)

            idx += 1

    return node_attributes, {'gender':gender_group, 'region':region_group, 'age':age_group}


def load_node_attributes_nba(attribute_file):
    node_attributes = {}
    country_group = {}
    age_group = {}

    with open(attribute_file, 'r') as fin:
        lines = fin.readlines()
        idx = 0
        for line in lines:
            if idx == 0:
                idx += 1
                continue
            eachline = line.strip().split(',')
            node_idx = int(idx-1)
            age = int(float(eachline[0]))
            country = int(eachline[1])
            #age = int(float(eachline[2]))
            node_attributes[node_idx] = (age, country)

            if age not in age_group:
                age_group[age] = []
            age_group[age].append(node_idx)

            if country not in country_group:
                country_group[country] = []
            country_group[country].append(node_idx)


            idx += 1

    return node_attributes, {'age':age_group, 'country':country_group}



# load edge for pokec data
# csv format: src_node, dst_node, edge_weight (binary: 1.0)
def load_adjacency_matrix_pokec(file, M):
    adj_mtx = np.zeros((M,M))
    # load the adjacency matrix of size MxM of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        idx = 0 
        for line in lines:
            if idx == 0: 
                idx += 1
                continue
            eachline = line.strip().split(',')
            scr_node = int(eachline[0])
            dst_node = int(eachline[1])
            weight = float(eachline[2])
            adj_mtx[scr_node, dst_node] = weight 
            adj_mtx[dst_node, scr_node] = weight 
            idx += 1
    
    return adj_mtx


# load sensitive attributes for movielens
# dat format: user_idx, gender, age, occupation
def load_user_attributes_movielens(file, M):
    #(gender, age, occupation)
    user_attributes = {}
    gender_dist = {'F':0, 'M':0}
    age_dist = {1:0, 18:0, 25:0, 35:0, 45:0, 50:0, 56:0}
    occupation_dist = {occup:0 for occup in range(21)}

    with open(file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            eachline = line.strip().split('::')
            user_idx = int(eachline[0])
            gender = eachline[1]
            age = int(eachline[2])
            occupation = int(eachline[3])
            user_attributes[user_idx] = (gender, age, occupation)

    return user_attributes


# load edges for movielens
# dat format: user_idx, item_idx, rating
def load_rating_matrix_movielens(file, M, N):
    over_rating_sparse_mtx = {}
    over_rating_mtx = np.zeros((M,N))
    #load the overall rating matrices of size MxN of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            eachline = line.strip().split('::')
            user_idx = int(eachline[0])
            item_idx = int(eachline[1])
            rating = int(eachline[2])
            over_rating_mtx[user_idx, item_idx] = rating
            over_rating_sparse_mtx[(user_idx, item_idx)] = rating
    
    return over_rating_sparse_mtx, over_rating_mtx

def eval_unbiasedness_twitter(data_name, embed_file=None):  # if file is none, then evalueate random embedding
    
    if data_name == 'twitter': M = 18470 
    node_attributes, attr_groups = load_node_attributes_twitter('{}/{}_node_attribute.csv'.format(DATA_FOLDER, data_name))
    adj_mtx = load_adjacency_matrix_twitter('{}/{}_edge.csv'.format(DATA_FOLDER, data_name), M)
    
    opinions = np.array([node_attributes[i] for i in range(M)])
    #regions = np.array([node_attributes[i][1] for i in range(M)])
    #ages = np.array([node_attributes[i][2] for i in range(M)])
    
    #print(genders)

    attribute_labels = {'opinion': opinions}
    
    
    if embed_file:  # learned embeddings loaded from valid file
        embedding = pk.load(open(embed_file, 'rb'))
    else: # random embeddings
        embedding = np.random.rand(*(M,16))
    
    results = {
        'unbiasedness': {
            'opinion': 0.0
        },
        'fairness-DP':{
            'opinion': 0.0
        },
        'fairness-EO':{
            'opinion': 0.0
        },
        'utility': 0.0
    }
    '''
    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'region']:
                
        # eval learned embedding
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            embedding[:10000], attribute_labels[evaluate_attr][:10000])
        pred = lgreg.predict(embedding[10000:])
        
        score = f1_score(attribute_labels[evaluate_attr][10000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score
    '''
        
    # eval fairness (DP & EO)
    print('Fairness evaluation (DP & EO)')
    for evaluate_attr in ['opinion']:
        
        if evaluate_attr == 'opinion': 
            num_sample_pairs = 200000
            attr_group = attr_groups[evaluate_attr]
            keys = list(attr_group.keys())
            for key in keys:
                if len(attr_group[key]) < 1000:
                    del attr_group[key]
        else:
            num_sample_pairs = 1000000
            attr_group = attr_groups[evaluate_attr]
            
        attr_values = list(attr_group.keys())
        num_attr_value = len(attr_values)
        #comb_indices = np.ndindex(*(num_attr_value,num_attr_value))


        Beta = [0,0.2,0.4,0.6,0.8,1]
        for beta in Beta:
          print("================this for beta=",beta,"===================")
          count = 0
          count1 = 0
          count_u = 0
          final_ndcg = 0
          final_ndcg_u = 0
          counter=0
          DP_list = []
          DP_algo = []
          EO_list = []
          comb_indices = np.ndindex(*(num_attr_value,num_attr_value))
          for comb_idx in comb_indices:
              group0 = attr_group[attr_values[comb_idx[0]]]
              group1 = attr_group[attr_values[comb_idx[1]]]
              group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
              group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)
              pair0_nodes = np.array([group0_nodes])
              pair1_nodes = np.array([group1_nodes])
              pair_scores = np.sum(np.multiply(embedding[group0_nodes], embedding[group1_nodes]), axis=1)
              # we now have to do argsort here take beta as input and out of total top beta fraction are selected (1) and remaing are (0) 
              # compute the DP based on these scores only, forget about the rest
             
              #pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              #pair_scores_algo = np.zeros(len(pair_scores))
              if count==0:
                group0_nodes_data = []
                group1_nodes_data = []
                for node_id in range(M):
                  s = random.sample(range(M),50)
                  for j in s:
                    group0_nodes_data.append(node_id)
                    group1_nodes_data.append(j)
                pair0_nodes_data = np.array([group0_nodes_data])
                pair1_nodes_data = np.array([group1_nodes_data])
                topic0 = np.array([opinions[group0_nodes_data]])
                topic1 = np.array([opinions[group1_nodes_data]])
                pair_scores_data = np.sum(np.multiply(embedding[group0_nodes_data], embedding[group1_nodes_data]), axis=1)
                scores = np.array([pair_scores_data])
                data = pd.DataFrame(np.concatenate((pair0_nodes_data.T, topic0.T, pair1_nodes_data.T, topic1.T, scores.T), axis = 1), columns=['node0','opinion0','node1','opinion1','score'])
                #data = pd.DataFrame(np.concatenate((pair0_nodes_data.T, topic0.T, pair1_nodes_data.T, topic1.T, scores.T), axis = 1), columns=['node0','topic0',>
                final = data.to_csv("alltwitter_opinion.csv", index = False, header= True)
                count = count+1

              epsilon = 0.01
              pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              pair_scores_algo = np.zeros(len(pair_scores))
              topbeta = int((1-beta)*len(pair_scores))
              topbeta_eps = int((1-beta+epsilon)*len(pair_scores)) 
              for i in range(len(pair_scores)):
                if pair_scores_sorted[i] >= topbeta:
                  if pair_scores_sorted[i] > topbeta_eps and counter>=0:
                    pair_scores_algo[i]=1
                  elif pair_scores_sorted[i] <= topbeta_eps and counter==0:
                    pair_scores_algo[i]=1
                else:
                  pair_scores_algo[i]=0  
              
              counter = counter+1


              DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
              #print(np.sum(sigmoid(pair_scores)), "numerator")
              #print(num_sample_pairs, "dennominator")
              DP_prob_algo = np.sum(pair_scores_algo) / num_sample_pairs
              DP_list.append(DP_prob)
              DP_algo.append(DP_prob_algo)

              comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
              if np.sum(comb_edge_indicator) > 0:
                  EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                  EO_list.append(EO_prob)
          
            
              k = 10
              accum_ndcg = 0
              node_cnt = 0
              accum_ndcg_u =0
              node_cnt_u = 0
              adj_mtx_fair = lil_matrix((M,M))
              adj_mtx_unfair = lil_matrix((M,M))
              selected_pairs = lil_matrix((M,M))
              unfair = sigmoid(pair_scores)
              #print("Hooha", pair0_nodes[0][2])
              n = num_sample_pairs
              for i in range(n):
                  adj_mtx_unfair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = unfair[i]
                  selected_pairs[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = 1
                  adj_mtx_fair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = pair_scores_algo[i]
              print('Utility evaluation (link prediction)')
              s = random.sample(range(M),5000)
              for node_id in s:
                  node_edges = adj_mtx[node_id]
                  test_pos_nodes = []
                  neg_nodes = []
                  for i in range(M):
                      if selected_pairs[node_id,i]==1:
                          if adj_mtx[node_id,i]>0:
                              test_pos_nodes.append(i)
                          else:
                              neg_nodes.append(i)
                  num_pos = len(test_pos_nodes)
                  all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes))
                  all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


                  all_eval_edges[:num_pos] = 1
                  edges = []
                  pred_edges_fair_pos = []
                  pred_edges_fair_neg = []

                  pred_edges_unfair_pos = []
                  pred_edges_unfair_neg = []

                  for i in range(M):
                      if int(selected_pairs[node_id,i])==1:
                          if adj_mtx[node_id,i]>0:
                              pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                          else:
                              pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])
                      pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
                      pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
                      if len(pred_edges_unfair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                          #print(rank_pred_keys)
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg_u = ndcg_at_k(ranked_node_edges, k)
                          if ndcg_u != 0.0:
                              accum_ndcg_u += ndcg_u
                              node_cnt_u += 1
                      if len(pred_edges_fair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg = ndcg_at_k(ranked_node_edges, k)
                          if ndcg != 0.0:
                              accum_ndcg += ndcg
                              node_cnt += 1
              if node_cnt != 0: 
                score = accum_ndcg/node_cnt
                count1 = count1 + 1
              else: 
                score=0
              if node_cnt_u != 0:
                score_u = accum_ndcg_u/node_cnt_u
                count_u = count_u + 1
              else: 
                score_u=0

              final_ndcg = final_ndcg + score
              final_ndcg_u = final_ndcg_u + score_u
              print('-- ndcg of link prediction for Algo score:{}'.format(score))
              print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
          if count1>0 and count_u>0:
            f = final_ndcg/count1
            f_u = final_ndcg_u/count_u
            print(DP_algo)
            print(DP_list)
            DP_value_algo = max(DP_algo) - min(DP_algo)    
            DP_value = max(DP_list) - min(DP_list)
            EO_value = max(EO_list) - min(EO_list)
            print('-- DP_value when predicting {}:{}'.format(evaluate_attr, DP_value))
            print('-- DP_value_algo when predicting {}:{}'.format(evaluate_attr, DP_value_algo))
            print('-- EO_value when predicting {}:{}'.format(evaluate_attr, EO_value))
            print('-- ndcg of link prediction for Algo score:{}'.format(f))
            print('-- ndcg of link prediction for unfair score:{}'.format(f_u))

        
        results['fairness-DP'][evaluate_attr] = DP_value
        results['fairness-EO'][evaluate_attr] = EO_value
                
    return results    

def eval_unbiasedness_nba(data_name, embed_file=None):  # if file is none, then evalueate random embedding
    
    if data_name == 'nba': M = 403 
    node_attributes, attr_groups = load_node_attributes_nba('{}/{}_node_attribute.csv'.format(DATA_FOLDER, data_name))
    adj_mtx = load_adjacency_matrix_nba('{}/{}_edge.csv'.format(DATA_FOLDER, data_name), M)

    ages = np.array([node_attributes[i][0] for i in range(M)])
    countries = np.array([node_attributes[i][1] for i in range(M)])
    #ages = np.array([node_attributes[i][2] for i in range(M)])
    maxi = np.max(ages)
    mini = np.min(ages)
    print(maxi,mini)
    print(np.unique(ages))

    attribute_labels = {'age': ages, 'country': countries}
    
    
    if embed_file:  # learned embeddings loaded from valid file
        embedding = pk.load(open(embed_file, 'rb'))
    else: # random embeddings
        embedding = np.random.rand(*(M,16))
    
    results = {
        'unbiasedness': {
            'age': 0.0, 
            'country': 0.0
        },
        'fairness-DP':{
            'age': 0.0, 
            'country': 0.0
        },
        'fairness-EO':{
            'age': 0.0, 
            'country': 0.0
        },
        'utility': 0.0
    }
    
    # eval micro-f1 for attribute prediction (unbiasedness)
    '''
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['age', 'country']:
                
        # eval learned embedding
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            embedding[:50000], attribute_labels[evaluate_attr][:50000])
        pred = lgreg.predict(embedding[50000:])
        
        score = f1_score(attribute_labels[evaluate_attr][50000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score
    '''
        
    # eval fairness (DP & EO)
    print('Fairness evaluation (DP & EO)')
    for evaluate_attr in ['country','age']:  #actually it is ['country','age']
        
        if evaluate_attr == 'age': 
            num_sample_pairs = 20000
            attr_group = attr_groups[evaluate_attr]
            keys = list(attr_group.keys())
            for key in keys:
                if len(attr_group[key]) < 8:
                    del attr_group[key]
        else:
            num_sample_pairs = 20000
            attr_group = attr_groups[evaluate_attr]
            
        attr_values = list(attr_group.keys())
        num_attr_value = len(attr_values)

        #f = open("alltwitter_opinion.csv", "w")
        #pair_scores = np.zeros(num_sample_pairs)
        #pair_scores_algo = np.zeros(num_sample_pairs) 
        #f.write("node0,gender0,age0,node1,gender1,age1,score\n")
        # the algo was take top beta most confident beta fraction of pairs in a particular group, and give the remaining pairs 
        # a random score. remaining 1-\beta fraction gets the same score as before
        Beta = [0,0.2,0.4,0.6,0.8,1]
        for beta in Beta:
          print("================this for beta=",beta,"===================")
          #count = 0
          count1 = 0
          count_u = 0
          final_ndcg = 0
          final_ndcg_u = 0
          counter=0
          DP_list = []
          DP_algo = []
          EO_list = []
          comb_indices = np.ndindex(*(num_attr_value,num_attr_value))
          for comb_idx in comb_indices:
              #print("came here as wel")
              group0 = attr_group[attr_values[comb_idx[0]]]
              group1 = attr_group[attr_values[comb_idx[1]]]
              group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
              group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)
              pair0_nodes = np.array([group0_nodes])
              pair1_nodes = np.array([group1_nodes])
              pair_scores = np.sum(np.multiply(embedding[group0_nodes], embedding[group1_nodes]), axis=1)
              # we now have to do argsort here take beta as input and out of total top beta fraction are selected (1) and remaing are (0) 
              # compute the DP based on these scores only, forget about the rest
             
              #pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              #pair_scores_algo = np.zeros(len(pair_scores))
              epsilon = 0.01
              pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              pair_scores_algo = np.zeros(len(pair_scores))
              topbeta = int((1-beta)*len(pair_scores))
              topbeta_eps = int((1-beta+epsilon)*len(pair_scores)) 
              for i in range(len(pair_scores)):
                if pair_scores_sorted[i] >= topbeta:
                  if pair_scores_sorted[i] > topbeta_eps and counter>=0:
                    pair_scores_algo[i]=1
                  elif pair_scores_sorted[i] <= topbeta_eps and counter==0:
                    pair_scores_algo[i]=1
                else:
                  pair_scores_algo[i]=0  
              
              counter = counter+1

            
              DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
              #print(np.sum(sigmoid(pair_scores)), "numerator")
              #print(num_sample_pairs, "dennominator")
              DP_prob_algo = np.sum(pair_scores_algo) / num_sample_pairs
              DP_list.append(DP_prob)
              DP_algo.append(DP_prob_algo)

              comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
              if np.sum(comb_edge_indicator) > 0:
                  EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                  EO_list.append(EO_prob)
          
            
              k = 10
              accum_ndcg = 0
              node_cnt = 0
              accum_ndcg_u =0
              node_cnt_u = 0
              adj_mtx_fair = lil_matrix((M,M))
              adj_mtx_unfair = lil_matrix((M,M))
              selected_pairs = lil_matrix((M,M))
              unfair = sigmoid(pair_scores)
              #print("Hooha", pair0_nodes[0][2])
              n = num_sample_pairs
              for i in range(n):
                  adj_mtx_unfair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = unfair[i]
                  selected_pairs[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = 1
                  adj_mtx_fair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = pair_scores_algo[i]
              #print('Utility evaluation (link prediction)')
              s = random.sample(range(M),400)
              for node_id in s:
                  node_edges = adj_mtx[node_id]
                  test_pos_nodes = []
                  neg_nodes = []
                  for i in range(M):
                      if selected_pairs[node_id,i]==1:
                          if adj_mtx[node_id,i]>0:
                              test_pos_nodes.append(i)
                          else:
                              neg_nodes.append(i)
                  num_pos = len(test_pos_nodes)
                  all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes))
                  all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


                  all_eval_edges[:num_pos] = 1
                  edges = []
                  pred_edges_fair_pos = []
                  pred_edges_fair_neg = []

                  pred_edges_unfair_pos = []
                  pred_edges_unfair_neg = []

                  for i in range(M):
                      if int(selected_pairs[node_id,i])==1:
                          if adj_mtx[node_id,i]>0:
                              pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                          else:
                              pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])
                      pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
                      pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
                      if len(pred_edges_unfair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                          #print(rank_pred_keys)
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg_u = ndcg_at_k(ranked_node_edges, k)
                          if ndcg_u != 0.0:
                              accum_ndcg_u += ndcg_u
                              node_cnt_u += 1
                      if len(pred_edges_fair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg = ndcg_at_k(ranked_node_edges, k)
                          if ndcg != 0.0:
                              accum_ndcg += ndcg
                              node_cnt += 1
              if node_cnt != 0: 
                score = accum_ndcg/node_cnt
                count1 = count1 + 1
              else: 
                score=0
              if node_cnt_u != 0:
                score_u = accum_ndcg_u/node_cnt_u
                count_u = count_u + 1
              else: 
                score_u=0

              final_ndcg = final_ndcg + score
              final_ndcg_u = final_ndcg_u + score_u
              #print('-- ndcg of link prediction for Algo score:{}'.format(score))
              #print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
          if count1>0 and count_u>0:
            f = final_ndcg/count1
            f_u = final_ndcg_u/count_u
            print(DP_algo)
            print(DP_list)
            DP_value_algo = max(DP_algo) - min(DP_algo)    
            DP_value = max(DP_list) - min(DP_list)
            EO_value = max(EO_list) - min(EO_list)
            print('-- DP_value when predicting {}:{}'.format(evaluate_attr, DP_value))
            print('-- DP_value_algo when predicting {}:{}'.format(evaluate_attr, DP_value_algo))
            print('-- EO_value when predicting {}:{}'.format(evaluate_attr, EO_value))
            print('-- ndcg of link prediction for Algo score:{}'.format(f))
            print('-- ndcg of link prediction for unfair score:{}'.format(f_u))
        

        results['fairness-DP'][evaluate_attr] = DP_value
        results['fairness-EO'][evaluate_attr] = EO_value
                
    return results    


def eval_unbiasedness_cora(data_name, embed_file=None):  # if file is none, then evalueate random embedding
    
    if data_name == 'cora': M = 2708 
    node_attributes, attr_groups = load_node_attributes_cora('{}/{}_node_attribute.csv'.format(DATA_FOLDER, data_name))
    adj_mtx = load_adjacency_matrix_nba('{}/{}_edge.csv'.format(DATA_FOLDER, data_name), M)
    
    topics = np.array([node_attributes[i] for i in range(M)])
    #countries = np.array([node_attributes[i][1] for i in range(M)])
    #ages = np.array([node_attributes[i][2] for i in range(M)])
    
    #print(genders)

    attribute_labels = {'topic': topics}
    
    
    if embed_file:  # learned embeddings loaded from valid file
        embedding = pk.load(open(embed_file, 'rb'))
    else: # random embeddings
        embedding = np.random.rand(*(M,16))
    
    results = {
        'unbiasedness': {
            'topic': 0.0
        },
        'fairness-DP':{
            'topic': 0.0
        },
        'fairness-EO':{
            'topic': 0.0 
        },
        'utility': 0.0
    }
    
    # eval micro-f1 for attribute prediction (unbiasedness)
    '''
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['age', 'country']:
                
        # eval learned embedding
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            embedding[:50000], attribute_labels[evaluate_attr][:50000])
        pred = lgreg.predict(embedding[50000:])
        
        score = f1_score(attribute_labels[evaluate_attr][50000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score
    '''
        
    # eval fairness (DP & EO)
    print('Fairness evaluation (DP & EO)')
    for evaluate_attr in ['topic']:  #actually it is ['country','age']
        
        if evaluate_attr == 'topic': 
            num_sample_pairs = 20000
            attr_group = attr_groups[evaluate_attr]
            keys = list(attr_group.keys())
            for key in keys:
                if len(attr_group[key]) < 8:
                    del attr_group[key]
        else:
            num_sample_pairs = 20000
            attr_group = attr_groups[evaluate_attr]
            
        attr_values = list(attr_group.keys())
        num_attr_value = len(attr_values)
        #comb_indices = np.ndindex(*(num_attr_value,num_attr_value))


        Beta = [0.8,1,0.2,0.4]
        for beta in Beta:
          print("================this for beta=",beta,"===================")
          count = 0
          count1 = 0
          count_u = 0
          final_ndcg = 0
          final_ndcg_u = 0
          counter=0
          DP_list = []
          DP_algo = []
          EO_list = []
          comb_indices = np.ndindex(*(num_attr_value,num_attr_value))
          for comb_idx in comb_indices:
              group0 = attr_group[attr_values[comb_idx[0]]]
              group1 = attr_group[attr_values[comb_idx[1]]]
              group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
              group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)
              pair0_nodes = np.array([group0_nodes])
              pair1_nodes = np.array([group1_nodes])
              pair_scores = np.sum(np.multiply(embedding[group0_nodes], embedding[group1_nodes]), axis=1)
              if count==0:
                group0_nodes_data = []
                group1_nodes_data = []
                for node_id in range(M):
                  s = random.sample(range(M),100)
                  for j in s:
                    group0_nodes_data.append(node_id)
                    group1_nodes_data.append(j)
                pair0_nodes_data = np.array([group0_nodes_data])
                pair1_nodes_data = np.array([group1_nodes_data])
                topic0 = np.array([topics[group0_nodes_data]])
                topic1 = np.array([topics[group1_nodes_data]])
                pair_scores_data = np.sum(np.multiply(embedding[group0_nodes_data], embedding[group1_nodes_data]), axis=1)
                scores = np.array([pair_scores_data])
                data = pd.DataFrame(np.concatenate((pair0_nodes_data.T, topic0.T, pair1_nodes_data.T, topic1.T, scores.T), axis = 1), columns=['node0','topic0','node1','topic1','score'])
                final = data.to_csv("allcora_topic.csv", index = False, header= True)
                count = count+1
              # we now have to do argsort here take beta as input and out of total top beta fraction are selected (1) and remaing are (0) 
              # compute the DP based on these scores only, forget about the rest
             
              #pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              #pair_scores_algo = np.zeros(len(pair_scores))
              epsilon = 0.01
              pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              pair_scores_algo = np.zeros(len(pair_scores))
              topbeta = int((1-beta)*len(pair_scores))
              topbeta_eps = int((1-beta+epsilon)*len(pair_scores)) 
              for i in range(len(pair_scores)):
                if pair_scores_sorted[i] >= topbeta:
                  if pair_scores_sorted[i] > topbeta_eps and counter>=0:
                    pair_scores_algo[i]=1
                  elif pair_scores_sorted[i] <= topbeta_eps and counter==0:
                    pair_scores_algo[i]=1
                else:
                  pair_scores_algo[i]=0  
              
              counter = counter+1


              DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
              #print(np.sum(sigmoid(pair_scores)), "numerator")
              #print(num_sample_pairs, "dennominator")
              DP_prob_algo = np.sum(pair_scores_algo) / num_sample_pairs
              DP_list.append(DP_prob)
              DP_algo.append(DP_prob_algo)

              comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
              if np.sum(comb_edge_indicator) > 0:
                  EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                  EO_list.append(EO_prob)
          
            
              k = 10
              accum_ndcg = 0
              node_cnt = 0
              accum_ndcg_u =0
              node_cnt_u = 0
              adj_mtx_fair = lil_matrix((M,M))
              adj_mtx_unfair = lil_matrix((M,M))
              selected_pairs = lil_matrix((M,M))
              unfair = sigmoid(pair_scores)
              #print("Hooha", pair0_nodes[0][2])
              n = num_sample_pairs
              for i in range(n):
                  adj_mtx_unfair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = unfair[i]
                  selected_pairs[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = 1
                  adj_mtx_fair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = pair_scores_algo[i]
              print('Utility evaluation (link prediction)')
              s = random.sample(range(M),1000)
              for node_id in s:
                  node_edges = adj_mtx[node_id]
                  test_pos_nodes = []
                  neg_nodes = []
                  for i in range(M):
                      if selected_pairs[node_id,i]==1:
                          if adj_mtx[node_id,i]>0:
                              test_pos_nodes.append(i)
                          else:
                              neg_nodes.append(i)
                  num_pos = len(test_pos_nodes)
                  all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes))
                  all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


                  all_eval_edges[:num_pos] = 1
                  edges = []
                  pred_edges_fair_pos = []
                  pred_edges_fair_neg = []

                  pred_edges_unfair_pos = []
                  pred_edges_unfair_neg = []

                  for i in range(M):
                      if int(selected_pairs[node_id,i])==1:
                          if adj_mtx[node_id,i]>0:
                              pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                          else:
                              pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])
                      pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
                      pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
                      if len(pred_edges_unfair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                          #print(rank_pred_keys)
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg_u = ndcg_at_k(ranked_node_edges, k)
                          if ndcg_u != 0.0:
                              accum_ndcg_u += ndcg_u
                              node_cnt_u += 1
                      if len(pred_edges_fair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg = ndcg_at_k(ranked_node_edges, k)
                          if ndcg != 0.0:
                              accum_ndcg += ndcg
                              node_cnt += 1
              if node_cnt != 0: 
                score = accum_ndcg/node_cnt
                count1 = count1 + 1
              else: 
                score=0
              if node_cnt_u != 0:
                score_u = accum_ndcg_u/node_cnt_u
                count_u = count_u + 1
              else: 
                score_u=0

              final_ndcg = final_ndcg + score
              final_ndcg_u = final_ndcg_u + score_u
              print('-- ndcg of link prediction for Algo score:{}'.format(score))
              print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
          if count1>0 and count_u>0:
            f = final_ndcg/count1
            f_u = final_ndcg_u/count_u
            print(DP_algo)
            print(DP_list)
            DP_value_algo = max(DP_algo) - min(DP_algo)    
            DP_value = max(DP_list) - min(DP_list)
            EO_value = max(EO_list) - min(EO_list)
            print('-- DP_value when predicting {}:{}'.format(evaluate_attr, DP_value))
            print('-- DP_value_algo when predicting {}:{}'.format(evaluate_attr, DP_value_algo))
            print('-- EO_value when predicting {}:{}'.format(evaluate_attr, EO_value))
            print('-- ndcg of link prediction for Algo score:{}'.format(f))
            print('-- ndcg of link prediction for unfair score:{}'.format(f_u))

        results['fairness-DP'][evaluate_attr] = DP_value
        results['fairness-EO'][evaluate_attr] = EO_value
                
    return results    




def eval_unbiasedness_dblp(data_name, embed_file=None):  # if file is none, then evalueate random embedding
    
    if data_name == 'dblp': M = 423469 
    node_attributes, attr_groups = load_node_attributes_dblp('{}/{}_node_attribute.csv'.format(DATA_FOLDER, data_name))
    adj_mtx = load_adjacency_matrix_dblp('{}/{}_edge.csv'.format(DATA_FOLDER, data_name), M)
    
    continents = np.array([node_attributes[i] for i in range(M)])
    #regions = np.array([node_attributes[i][1] for i in range(M)])
    #ages = np.array([node_attributes[i][2] for i in range(M)])
    
    #print(genders)

    attribute_labels = {'continent': continents}
    
    
    if embed_file:  # learned embeddings loaded from valid file
        embedding = pk.load(open(embed_file, 'rb'))
    else: # random embeddings
        embedding = np.random.rand(*(M,16))
    
    results = {
        'unbiasedness': {
            'continent': 0.0
        },
        'fairness-DP':{
            'continent': 0.0
        },
        'fairness-EO':{
            'continent': 0.0
        },
        'utility': 0.0
    }
    '''
    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'region']:
                
        # eval learned embedding
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            embedding[:10000], attribute_labels[evaluate_attr][:10000])
        pred = lgreg.predict(embedding[10000:])
        
        score = f1_score(attribute_labels[evaluate_attr][10000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score
    '''
        
    # eval fairness (DP & EO)
    print('Fairness evaluation (DP & EO)')
    for evaluate_attr in ['continent']:
        
        if evaluate_attr == 'continent': 
            num_sample_pairs = 200000
            attr_group = attr_groups[evaluate_attr]
            keys = list(attr_group.keys())
            for key in keys:
                if len(attr_group[key]) < 1000:
                    del attr_group[key]
        else:
            num_sample_pairs = 1000000
            attr_group = attr_groups[evaluate_attr]
            
        attr_values = list(attr_group.keys())
        num_attr_value = len(attr_values)
        comb_indices = np.ndindex(*(num_attr_value,num_attr_value))

        DP_list = []
        DP_algo = []
        EO_list = []
        #f = open("alltwitter_opinion.csv", "w")
        #pair_scores = np.zeros(num_sample_pairs)
        #pair_scores_algo = np.zeros(num_sample_pairs) 
        n = num_sample_pairs
        #f.write("node0,gender0,age0,node1,gender1,age1,score\n")
        # the algo was take top beta most confident beta fraction of pairs in a particular group, and give the remaining pairs 
        # a random score. remaining 1-\beta fraction gets the same score as before
        count = 0
        for comb_idx in comb_indices:
            group0 = attr_group[attr_values[comb_idx[0]]]
            group1 = attr_group[attr_values[comb_idx[1]]]
            group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
            group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)
            pair0_nodes = np.array([group0_nodes])
            pair1_nodes = np.array([group1_nodes])    

            pair_scores = np.sum(np.multiply(embedding[group0_nodes], embedding[group1_nodes]), axis=1)
            print(pair0_nodes) 
            # we now have to do argsort here take beta as input and out of total top beta fraction are selected (1) and remaing are (0) 
            # compute the DP based on these scores only, forget about the rest
            beta=0.3
            pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
            topbeta = int((1-beta)*len(pair_scores))
            pair_scores_algo = np.zeros(len(pair_scores))

            for i in range(len(pair_scores)):
             if pair_scores_sorted[i] >= topbeta:
              pair_scores_algo[i]=1
             else:
              pair_scores_algo[i]=0  



            DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
            
            DP_prob_algo = np.sum(pair_scores_algo) / num_sample_pairs
            DP_list.append(DP_prob)
            DP_algo.append(DP_prob_algo)

            comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
            if np.sum(comb_edge_indicator) > 0:
                EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                EO_list.append(EO_prob)
            count = count + 1
        
          
            k = 10
            accum_ndcg = 0
            node_cnt = 0
            accum_ndcg_u =0
            node_cnt_u = 0
            adj_mtx_fair = lil_matrix((M,M))
            adj_mtx_unfair = lil_matrix((M,M))
            selected_pairs = lil_matrix((M,M))
            unfair = sigmoid(pair_scores)
            print("Hooha", pair0_nodes[0][2])
            for i in range(n):
                adj_mtx_unfair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = unfair[i]
                selected_pairs[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = 1
                adj_mtx_fair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = pair_scores_algo[i]
            print('Utility evaluation (link prediction)')
            s = random.sample(range(M),5000)
            for node_id in s:
                node_edges = adj_mtx[node_id]
                test_pos_nodes = []
                neg_nodes = []
                for i in range(M):
                    if selected_pairs[node_id,i]==1:
                        if adj_mtx[node_id,i]>0:
                            test_pos_nodes.append(i)
                        else:
                            neg_nodes.append(i)
                num_pos = len(test_pos_nodes)
                all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes))
                all_eval_edges = np.zeros(100)  # because each node has 20 neighbors in the set


                all_eval_edges[:num_pos] = 1
                edges = []
                pred_edges_fair_pos = []
                pred_edges_fair_neg = []

                pred_edges_unfair_pos = []
                pred_edges_unfair_neg = []

                for i in range(M):
                    if int(selected_pairs[node_id,i])==1:
                        if adj_mtx[node_id,i]>0:
                            pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                            pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                        else:
                            pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                            pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])
                    pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
                    pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
                    if len(pred_edges_unfair) >=k:
                        rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                        #print(rank_pred_keys)
                        ranked_node_edges = all_eval_edges[rank_pred_keys]
                        ndcg_u = ndcg_at_k(ranked_node_edges, k)
                        if ndcg_u != 0.0:
                            accum_ndcg_u += ndcg_u
                            node_cnt_u += 1
                    if len(pred_edges_fair) >=k:
                        rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                        ranked_node_edges = all_eval_edges[rank_pred_keys]
                        ndcg = ndcg_at_k(ranked_node_edges, k)
                        if ndcg != 0.0:
                            accum_ndcg += ndcg
                            node_cnt += 1
            score = accum_ndcg/node_cnt
            score_u = accum_ndcg_u/node_cnt_u
            print('-- ndcg of link prediction for Algo score:{}'.format(score))
            print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
        
        print(DP_algo)
        print(DP_list)
        DP_value_algo = max(DP_algo) - min(DP_algo)    
        DP_value = max(DP_list) - min(DP_list)
        EO_value = max(EO_list) - min(EO_list)
        print('-- DP_value when predicting {}:{}'.format(evaluate_attr, DP_value))
        print('-- DP_value_algo when predicting {}:{}'.format(evaluate_attr, DP_value_algo))
        print('-- EO_value when predicting {}:{}'.format(evaluate_attr, EO_value))
        
        results['fairness-DP'][evaluate_attr] = DP_value
        results['fairness-EO'][evaluate_attr] = EO_value
                
    return results    



def eval_unbiasedness_polblog(data_name, embed_file=None):  # if file is none, then evalueate random embedding
    
    if data_name == 'pol-blog': M = 1222 
    node_attributes, attr_groups = load_node_attributes_polblog('{}/{}_node_attribute.csv'.format(DATA_FOLDER, data_name))
    adj_mtx = load_adjacency_matrix_polblog('{}/{}_edge.csv'.format(DATA_FOLDER, data_name), M)
    
    parties = np.array([node_attributes[i] for i in range(M)])
    #regions = np.array([node_attributes[i][1] for i in range(M)])
    #ages = np.array([node_attributes[i][2] for i in range(M)])
    
    #print(genders)

    attribute_labels = {'party': parties}
    
    
    if embed_file:  # learned embeddings loaded from valid file
        embedding = pk.load(open(embed_file, 'rb'))
    else: # random embeddings
        embedding = np.random.rand(*(M,16))
    
    results = {
        'unbiasedness': {
            'party': 0.0
        },
        'fairness-DP':{
            'party': 0.0
        },
        'fairness-EO':{
            'party': 0.0
        },
        'utility': 0.0
    }
    '''
    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'region']:
                
        # eval learned embedding
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            embedding[:10000], attribute_labels[evaluate_attr][:10000])
        pred = lgreg.predict(embedding[10000:])
        
        score = f1_score(attribute_labels[evaluate_attr][10000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score
    '''
        
    # eval fairness (DP & EO)
    print('Fairness evaluation (DP & EO)')
    for evaluate_attr in ['party']:
        
        if evaluate_attr == 'party': 
            num_sample_pairs = 200000
            attr_group = attr_groups[evaluate_attr]
            keys = list(attr_group.keys())
            for key in keys:
                if len(attr_group[key]) < 10:
                    del attr_group[key]
        else:
            num_sample_pairs = 1000000
            attr_group = attr_groups[evaluate_attr]
            
        attr_values = list(attr_group.keys())
        print("attr_values", attr_values)
        num_attr_value = len(attr_values)
        print("num", num_attr_value)
        comb_indices = np.ndindex(*(num_attr_value,num_attr_value))
        print("comb_indices are",comb_indices)

        Beta = [0,0.2,0.4,0.6,0.8,1]
        for beta in Beta:
          print("================this for beta=",beta,"===================")
          count = 0
          count1 = 0
          count_u = 0
          final_ndcg = 0
          final_ndcg_u = 0
          counter=0
          DP_list = []
          DP_algo = []
          EO_list = []
          comb_indices = np.ndindex(*(num_attr_value,num_attr_value))
          for comb_idx in comb_indices:
              group0 = attr_group[attr_values[comb_idx[0]]]
              group1 = attr_group[attr_values[comb_idx[1]]]
              group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
              group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)
              pair0_nodes = np.array([group0_nodes])
              pair1_nodes = np.array([group1_nodes])
              pair_scores = np.sum(np.multiply(embedding[group0_nodes], embedding[group1_nodes]), axis=1)
              if count==0:
                group0_nodes_data = []
                group1_nodes_data = []
                for node_id in range(M):
                  s = random.sample(range(M),100)
                  for j in s:
                    group0_nodes_data.append(node_id)
                    group1_nodes_data.append(j)
                pair0_nodes_data = np.array([group0_nodes_data])
                pair1_nodes_data = np.array([group1_nodes_data])
                party0 = np.array([parties[group0_nodes_data]])
                party1 = np.array([parties[group1_nodes_data]])
                pair_scores_data = np.sum(np.multiply(embedding[group0_nodes_data], embedding[group1_nodes_data]), axis=1)
                scores = np.array([pair_scores_data])
                data = pd.DataFrame(np.concatenate((pair0_nodes_data.T, party0.T, pair1_nodes_data.T, party1.T, scores.T), axis = 1), columns=['node0','party0','node1','party1','score'])
                #data = pd.DataFrame(np.concatenate((pair0_nodes_data.T, party0.T, pair1_nodes_data.T, party1.T, scores.T), axis = 1), columns=['node0','topic0',>
                final = data.to_csv("allpolblog_party.csv", index = False, header= True)
                count = count+1

              # we now have to do argsort here take beta as input and out of total top beta fraction are selected (1) and remaing are (0) 
              # compute the DP based on these scores only, forget about the rest
             
              #pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              #pair_scores_algo = np.zeros(len(pair_scores))
              epsilon = 0.01
              pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              pair_scores_algo = np.zeros(len(pair_scores))
              topbeta = int((1-beta)*len(pair_scores))
              topbeta_eps = int((1-beta+epsilon)*len(pair_scores)) 
              for i in range(len(pair_scores)):
                if pair_scores_sorted[i] >= topbeta:
                  if pair_scores_sorted[i] > topbeta_eps and counter>=0:
                    pair_scores_algo[i]=1
                  elif pair_scores_sorted[i] <= topbeta_eps and counter==0:
                    pair_scores_algo[i]=1
                else:
                  pair_scores_algo[i]=0  
              
              counter = counter+1


              DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
              #print(np.sum(sigmoid(pair_scores)), "numerator")
              #print(num_sample_pairs, "dennominator")
              DP_prob_algo = np.sum(pair_scores_algo) / num_sample_pairs
              DP_list.append(DP_prob)
              DP_algo.append(DP_prob_algo)

              comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
              if np.sum(comb_edge_indicator) > 0:
                  EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                  EO_list.append(EO_prob)
          
            
              k = 10
              accum_ndcg = 0
              node_cnt = 0
              accum_ndcg_u =0
              node_cnt_u = 0
              adj_mtx_fair = lil_matrix((M,M))
              adj_mtx_unfair = lil_matrix((M,M))
              selected_pairs = lil_matrix((M,M))
              unfair = sigmoid(pair_scores)
              #print("Hooha", pair0_nodes[0][2])
              n = num_sample_pairs
              for i in range(n):
                  adj_mtx_unfair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = unfair[i]
                  selected_pairs[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = 1
                  adj_mtx_fair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = pair_scores_algo[i]
              print('Utility evaluation (link prediction)')
              s = random.sample(range(M),800)
              for node_id in s:
                  node_edges = adj_mtx[node_id]
                  test_pos_nodes = []
                  neg_nodes = []
                  for i in range(M):
                      if selected_pairs[node_id,i]==1:
                          if adj_mtx[node_id,i]>0:
                              test_pos_nodes.append(i)
                          else:
                              neg_nodes.append(i)
                  num_pos = len(test_pos_nodes)
                  all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes))
                  all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


                  all_eval_edges[:num_pos] = 1
                  edges = []
                  pred_edges_fair_pos = []
                  pred_edges_fair_neg = []

                  pred_edges_unfair_pos = []
                  pred_edges_unfair_neg = []

                  for i in range(M):
                      if int(selected_pairs[node_id,i])==1:
                          if adj_mtx[node_id,i]>0:
                              pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                          else:
                              pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])
                      pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
                      pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
                      if len(pred_edges_unfair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                          #print(rank_pred_keys)
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg_u = ndcg_at_k(ranked_node_edges, k)
                          if ndcg_u != 0.0:
                              accum_ndcg_u += ndcg_u
                              node_cnt_u += 1
                      if len(pred_edges_fair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg = ndcg_at_k(ranked_node_edges, k)
                          if ndcg != 0.0:
                              accum_ndcg += ndcg
                              node_cnt += 1
              if node_cnt != 0: 
                score = accum_ndcg/node_cnt
                count1 = count1 + 1
              else: 
                score=0
              if node_cnt_u != 0:
                score_u = accum_ndcg_u/node_cnt_u
                count_u = count_u + 1
              else: 
                score_u=0

              final_ndcg = final_ndcg + score
              final_ndcg_u = final_ndcg_u + score_u
              print('-- ndcg of link prediction for Algo score:{}'.format(score))
              print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
          if count1>0 and count_u>0:
            f = final_ndcg/count1
            f_u = final_ndcg_u/count_u
            print(DP_algo)
            print(DP_list)
            DP_value_algo = max(DP_algo) - min(DP_algo)    
            DP_value = max(DP_list) - min(DP_list)
            EO_value = max(EO_list) - min(EO_list)
            print('-- DP_value when predicting {}:{}'.format(evaluate_attr, DP_value))
            print('-- DP_value_algo when predicting {}:{}'.format(evaluate_attr, DP_value_algo))
            print('-- EO_value when predicting {}:{}'.format(evaluate_attr, EO_value))
            print('-- ndcg of link prediction for Algo score:{}'.format(f))
            print('-- ndcg of link prediction for unfair score:{}'.format(f_u))

        
        results['fairness-DP'][evaluate_attr] = DP_value
        results['fairness-EO'][evaluate_attr] = EO_value
                
    return results    


def eval_unbiasedness_pokec(data_name, embed_file=None):  # if file is none, then evalueate random embedding
    
    if data_name == 'pokec-z': M = 67796 
    elif data_name == 'pokec-n': M = 66569 
    else: raise Exception('Invalid dataset!')
    
    node_attributes, attr_groups = load_node_attributes_pokec('{}/{}_node_attribute.csv'.format(DATA_FOLDER, data_name))
    adj_mtx = load_adjacency_matrix_pokec('{}/{}_edge.csv'.format(DATA_FOLDER, data_name), M)
    
    genders = np.array([node_attributes[i][0] for i in range(M)])
    regions = np.array([node_attributes[i][1] for i in range(M)])
    ages = np.array([node_attributes[i][2] for i in range(M)])
    
    print(genders)

    attribute_labels = {'gender': genders, 'age': ages, 'region': regions}
    
    
    if embed_file:  # learned embeddings loaded from valid file
        embedding = pk.load(open(embed_file, 'rb'))
    else: # random embeddings
        embedding = np.random.rand(*(M,16))
    
    results = {
        'unbiasedness': {
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        'fairness-DP':{
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        'fairness-EO':{
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        'utility': 0.0
    }
    
    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'region']:
                
        # eval learned embedding
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            embedding[:50000], attribute_labels[evaluate_attr][:50000])
        pred = lgreg.predict(embedding[50000:])
        
        score = f1_score(attribute_labels[evaluate_attr][50000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score

        
    # eval fairness (DP & EO)
    print('Fairness evaluation (DP & EO)')
    for evaluate_attr in ['age', 'region', 'gender']:
        
        if evaluate_attr == 'age': 
            num_sample_pairs = 200000
            attr_group = attr_groups[evaluate_attr]
            keys = list(attr_group.keys())
            for key in keys:
                if len(attr_group[key]) < 1000:
                    del attr_group[key]
        else:
            num_sample_pairs = 1000000
            attr_group = attr_groups[evaluate_attr]
            
        attr_values = list(attr_group.keys())
        num_attr_value = len(attr_values)
        #comb_indices = np.ndindex(*(num_attr_value,num_attr_value))



        Beta = [0,0.2,0.4,0.6,0.8,1]
        for beta in Beta:
          print("================this for beta=",beta,"===================")
          count = 0
          count1 = 0
          count_u = 0
          final_ndcg = 0
          final_ndcg_u = 0
          counter=0
          DP_list = []
          DP_algo = []
          EO_list = []
          comb_indices = np.ndindex(*(num_attr_value,num_attr_value))
          for comb_idx in comb_indices:
              group0 = attr_group[attr_values[comb_idx[0]]]
              group1 = attr_group[attr_values[comb_idx[1]]]
              group0_nodes = np.random.choice(group0, num_sample_pairs, replace=True)
              group1_nodes = np.random.choice(group1, num_sample_pairs, replace=True)
              pair0_nodes = np.array([group0_nodes])
              pair1_nodes = np.array([group1_nodes])
              pair_scores = np.sum(np.multiply(embedding[group0_nodes], embedding[group1_nodes]), axis=1)
              # we now have to do argsort here take beta as input and out of total top beta fraction are selected (1) and remaing are (0) 
              # compute the DP based on these scores only, forget about the rest
             
              #pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              #pair_scores_algo = np.zeros(len(pair_scores))
              epsilon = 0.01
              pair_scores_sorted = np.argsort(np.argsort(sigmoid(pair_scores)))
              pair_scores_algo = np.zeros(len(pair_scores))
              topbeta = int((1-beta)*len(pair_scores))
              topbeta_eps = int((1-beta+epsilon)*len(pair_scores)) 
              for i in range(len(pair_scores)):
                if pair_scores_sorted[i] >= topbeta:
                  if pair_scores_sorted[i] > topbeta_eps and counter>=0:
                    pair_scores_algo[i]=1
                  elif pair_scores_sorted[i] <= topbeta_eps and counter==0:
                    pair_scores_algo[i]=1
                else:
                  pair_scores_algo[i]=0  
              
              counter = counter+1


              DP_prob = np.sum(sigmoid(pair_scores)) / num_sample_pairs
              #print(np.sum(sigmoid(pair_scores)), "numerator")
              #print(num_sample_pairs, "dennominator")
              DP_prob_algo = np.sum(pair_scores_algo) / num_sample_pairs
              DP_list.append(DP_prob)
              DP_algo.append(DP_prob_algo)

              comb_edge_indicator = (adj_mtx[group0_nodes, group1_nodes] > 0).astype(int)
              if np.sum(comb_edge_indicator) > 0:
                  EO_prob = np.sum(sigmoid(pair_scores.T[0]) * comb_edge_indicator) / np.sum(comb_edge_indicator)
                  EO_list.append(EO_prob)
          
            
              k = 10
              accum_ndcg = 0
              node_cnt = 0
              accum_ndcg_u =0
              node_cnt_u = 0
              adj_mtx_fair = lil_matrix((M,M))
              adj_mtx_unfair = lil_matrix((M,M))
              selected_pairs = lil_matrix((M,M))
              unfair = sigmoid(pair_scores)
              #print("Hooha", pair0_nodes[0][2])
              n = num_sample_pairs
              for i in range(n):
                  adj_mtx_unfair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = unfair[i]
                  selected_pairs[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = 1
                  adj_mtx_fair[int(pair0_nodes[0][i]),int(pair1_nodes[0][i])] = pair_scores_algo[i]
              print('Utility evaluation (link prediction)')
              s = random.sample(range(M),1000)
              for node_id in s:
                  node_edges = adj_mtx[node_id]
                  test_pos_nodes = []
                  neg_nodes = []
                  for i in range(M):
                      if selected_pairs[node_id,i]==1:
                          if adj_mtx[node_id,i]>0:
                              test_pos_nodes.append(i)
                          else:
                              neg_nodes.append(i)
                  num_pos = len(test_pos_nodes)
                  all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes))
                  all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


                  all_eval_edges[:num_pos] = 1
                  edges = []
                  pred_edges_fair_pos = []
                  pred_edges_fair_neg = []

                  pred_edges_unfair_pos = []
                  pred_edges_unfair_neg = []

                  for i in range(M):
                      if int(selected_pairs[node_id,i])==1:
                          if adj_mtx[node_id,i]>0:
                              pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                          else:
                              pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                              pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])
                      pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
                      pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
                      if len(pred_edges_unfair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                          #print(rank_pred_keys)
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg_u = ndcg_at_k(ranked_node_edges, k)
                          if ndcg_u != 0.0:
                              accum_ndcg_u += ndcg_u
                              node_cnt_u += 1
                      if len(pred_edges_fair) >=k:
                          rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                          ranked_node_edges = all_eval_edges[rank_pred_keys]
                          ndcg = ndcg_at_k(ranked_node_edges, k)
                          if ndcg != 0.0:
                              accum_ndcg += ndcg
                              node_cnt += 1
              if node_cnt != 0: 
                score = accum_ndcg/node_cnt
                count1 = count1 + 1
              else: 
                score=0
              if node_cnt_u != 0:
                score_u = accum_ndcg_u/node_cnt_u
                count_u = count_u + 1
              else: 
                score_u=0

              final_ndcg = final_ndcg + score
              final_ndcg_u = final_ndcg_u + score_u
              print('-- ndcg of link prediction for Algo score:{}'.format(score))
              print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
          if count1>0 and count_u>0:
            f = final_ndcg/count1
            f_u = final_ndcg_u/count_u
            print(DP_algo)
            print(DP_list)
            DP_value_algo = max(DP_algo) - min(DP_algo)    
            DP_value = max(DP_list) - min(DP_list)
            EO_value = max(EO_list) - min(EO_list)
            print('-- DP_value when predicting {}:{}'.format(evaluate_attr, DP_value))
            print('-- DP_value_algo when predicting {}:{}'.format(evaluate_attr, DP_value_algo))
            print('-- EO_value when predicting {}:{}'.format(evaluate_attr, EO_value))
            print('-- ndcg of link prediction for Algo score:{}'.format(f))
            print('-- ndcg of link prediction for unfair score:{}'.format(f_u))

        results['fairness-DP'][evaluate_attr] = DP_value
        results['fairness-EO'][evaluate_attr] = EO_value
                
    return results 
#here only I would write the LP code
#what do I need for LP, scores and beta (may be)
#no sorting no nothing


##def LPG(data, scores):  #do for both intersectional and independent
# only thing needed is sensitive data info
# i.e. (gender,age,gender,age) for every pair and corresponding P_uv
# I have to first group them into interesectional groups
# like we did before
def eval_unbiasedness_movielens(data_name, embed_file=None):
    
    M = 6040 + 1
    N = 3952 + 1
        
    rating_sparse_mtx, rating_mtx = load_rating_matrix_movielens('{}/ml-1m/ratings.dat'.format(RAW_FOLDER), M, N)
    user_attributes = load_user_attributes_movielens('{}/ml-1m/users.dat'.format(RAW_FOLDER), M)

    genders = np.array([int(user_attributes[i][0]=='M') for i in range(1, M)])
    ages = np.array([int(user_attributes[i][1]) for i in range(1, M)])
    occupations = np.array([int(user_attributes[i][2]) for i in range(1, M)])

    attribute_labels = {'gender': genders, 'age': ages, 'occupation': occupations}

    rating_mtx = rating_mtx[1:]
    rating_mtx = rating_mtx[:,1:]
    
    if embed_file:
        unbiased_embedding = pk.load(open(embed_file,'rb')) 
        X = unbiased_embedding[:M-1]  # users
        Y = unbiased_embedding[M-1:]  # items
    else:
        X, Y = np.random.rand(*(M-1,16)), np.random.rand(*(N-1,16))
    
    results = {
        'unbiasedness': {
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        # 'fairness-DP':{
        #     'gender': 0.0, 
        #     'age': 0.0, 
        #     'region': 0.0
        # },
        # 'fairness-EO':{
        #     'gender': 0.0, 
        #     'age': 0.0, 
        #     'region': 0.0
        # },
        'utility': 0.0
    }

    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'occupation']:
        
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000).fit(
            X[:5000], attribute_labels[evaluate_attr][:5000])
        pred = lgreg.predict(X[5000:])

        # rating_lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000).fit(
        #     rating_mtx[:5000], attribute_labels[evaluate_attr][:5000])
        # rating_pred = rating_lgreg.predict(rating_mtx[5000:])
        
        score = f1_score(attribute_labels[evaluate_attr][5000:], pred, average='micro')
        
        print('-- micro-f1 when predicting {}: {}'.format(evaluate_attr, score))
        results['unbiasedness'][evaluate_attr] = score
        
        # score = f1_score(attribute_labels[evaluate_attr][5000:], rating_pred, average='micro')
        # print('-- raw rating micro-f1: ', score)


    #evaluate NDCG
    k = 10
    accum_ndcg = 0
    
    print('Utility evaluation (link prediction)')
    for user_id in range(1, M):
        user = user_id - 1
        user_ratings = rating_mtx[user] # (rating_mtx[user] > 0).astype(int)
        
        pred_ratings = np.dot(X[user], Y.T)
        rank_pred_keys = np.argsort(pred_ratings)[::-1]
        ranked_user_ratings = user_ratings[rank_pred_keys]
        ndcg = ndcg_at_k(ranked_user_ratings, k)
        accum_ndcg += ndcg
        
    score = accum_ndcg/M
    print('-- ndcg of link prediction:{}'.format(score))
    results['utility'] = score

    return results


# evalute embeddings by batch of files
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embedding_folder', type=str, default="./embeddings", help='embedding folder path')
    parser.add_argument('--dataset', type=str, default="pokec-z", help='dataset to evaluate')

    args = parse
