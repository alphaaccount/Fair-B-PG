import numpy as np
import pandas as pd
import pulp as p 
import math
import random
import cvxpy as cp
import networkx as nx
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


def loss(z):
    return sum((z-b)**2.0)


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
        #print('.', end=' ')
        return 0.
    return dcg_at_k(r, k) / idcg




def LPG(data,adj_mtx, epsilon):     #here we are taking two sensitive attrib # make the data here  
    m = 40 # total 40 groups are there 
    n = len(data)
    data1 = np.zeros((m,n))
    sizes_edges = np.zeros(m)

    adamic_adar = np.zeros(n, dtype=float)

    M = adj_mtx.shape[0]
    nbr = np.zeros(M, dtype=int)
   
    G = nx.from_numpy_matrix(adj_mtx)
   

    for i in range(M):
      for j in range(M):
        if adj_mtx[i,j] > 0:
          nbr[i] = nbr[i] + 1
   
    for i in range(n):
      u = int(data.iloc[i,0])
      v = int(data.iloc[i,3])
      for j in list(set(G[u]) & set(G[v])):
        adamic_adar[i] = adamic_adar[i] + 1/math.log2(nbr[j])
   
    a_adar_gr = np.zeros(m, dtype=float)

    sizes=np.zeros(m,dtype=int)
#     report_index(index,data1,e):  


    for i in range(len(data)):
        for j in range(2):
          for k in range(2):
            if data.iloc[i,1]==j and data.iloc[i,4]==k :
              data1[2*j+k][i] = 1
              a_adar_gr[2*j+k] =  a_adar_gr[2*j+k] + adamic_adar[i]
            if data.iloc[i,1]==j and data.iloc[i,4]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,3])] > 0:
              sizes_edges[2*j+k] = sizes_edges[2*j+k]+1
             
        for j in range(6):
          for k in range(6):
            if data.iloc[i,2]==j and data.iloc[i,5]==k :
              data1[6*j+k+4][i] = 1   #begins at 4
              a_adar_gr[6*j+k+4] =  a_adar_gr[6*j+k+4] + adamic_adar[i]
            if data.iloc[i,2]==j and data.iloc[i,5]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,3])] > 0:
              sizes_edges[6*j+k+4] = sizes_edges[6*j+k+4]+1


    max_size=0
    for i in range(m):
        count=0
        for j in range(n):
            if data1[i][j]==1:
                count=count+1 
        if count>max_size:
            max_size=count
        sizes[i]=count
    print(sizes)    


    print("data is")
    print(data1)
    print(m,n)

    
    #print(sizes_edges)
    #print(a_adar_gr/sizes_edges)
    print(sizes_edges/sizes)
    edge_density = np.zeros(m)
    for i in range(m):
      if sizes[i]==0:
        edge_density[i] = 0
      else:
        edge_density[i] = sizes_edges[i]*400000/sizes[i]
       
    ############### #  SORTED for ACCURACY ONLY ####
    m=40
    h1=[]
    key1=[]
    cost=np.zeros(n,dtype=int)
    data2=np.zeros((m,n),dtype=int)




    '''
    for i in range(n):
            h1.append(e[i][1])
            key1.append(i)

        
#print(hc)
#     print(key1)
    
    for i in range(1,len(h1)):
        for j in range(i,0,-1):
            var=0
            var2=0
            if h1[j-1]<h1[j]:
                index=j
                var=h1[j]
                h1[j]=h1[j-1]
                h1[j-1]=var

                var2=key1[j]
                key1[j]=key1[j-1]
                key1[j-1]=var2
            else:
                break
    

    
    
    for j in range(len(key1)):    
         data2[0][key1[j]]=j+1
    
    for j in range(n):
        summ=0
        summ=summ+data2[0][j] 
        cost[j]=summ
    '''  
    Lp_prob = p.LpProblem('Problem', p.LpMaximize)  
    solver = p.getSolver('PULP_CBC_CMD', timeLimit=20)
   
    
#     X=np.zeros(n+1,dtype=p.LpVariable)
    X=np.zeros(n+m+1,dtype=p.LpVariable)
    Y=np.zeros(m,dtype=p.LpVariable)
    
    #############################33
    
    beta_avg = 0.5
    beta_actual = []
    for i in range(m):
      beta_actual.append(beta_avg)
      
    ###############################
    #beta_actual = [0.30355010313755293, 0.10743801652892562, 0.252269224182223, 0.13278688524590163, 0.3118811881188119, 0.1, 0.13043478260869565]
    
   
    

    select_sizes=np.zeros(m,dtype=int)
   
    size_final=np.zeros(m,dtype=int)

    for i in range(m):
        var1 = str(n+100+i)
        Y[i]=p.LpVariable(var1,lowBound=0,upBound=1,cat='Continuous')
    
    for i in range(n):
        var1=str(i)       
        X[i]=p.LpVariable(var1,lowBound=0,upBound=1,cat='Continuous')
   
    X[n]=p.LpVariable(str(n),lowBound=0,upBound=1,cat='Continuous')  

    #tpr = p.LpVariable(str(n+200),lowBound=0,upBound=1,cat='Continuous')  
    #fpr = p.LpVariable(str(n+201),lowBound=0,upBound=1,cat='Continuous')  

#     for i in range(m):
#         k=n+i+1
#         alpha=(((sizes[i])*(sizes[i]+1))/2)
#         X[i]=p.LpVariable(var1,lowBound=(((beta*sizes[i])*(beta*sizes[i]+1))/2),upBound=alpha,cat='Continuous')
    
        
#     X[n]=  p.LpVariable("z1",lowBound=0)
    #X[n+1]=  p.LpVariable("z2",lowBound=0)
    group = np.zeros(n)
    for i in range(n):
      for j in range(2):
        for k in range(2):
          if data.iloc[i,1] == j and data.iloc[i,4] == k:
            group[i] = group[i] + edge_density[2*j+k]
     
      for j in range(6):
        for k in range(6):
          if data.iloc[i,2]==j and data.iloc[i,5]==k :
            group[i] = group[i] + edge_density[6*j+k+4]
            
    #########objective function#####################
    
#     Lp_prob += 2*X[n+1]+10*X[n+2]+9*X[n+3]+3*X[n+4]
    #alpha=0.8
    #beta_avg = 0.10
    Lp_prob+= p.lpSum([(X[j])*sigmoid(data.iloc[j,6])*group[j] for j in range(n)]) 
    #Lp_prob+=1      #Lp_prob += Y[0]*sizes[0] + Y[1]*sizes[1] >= p.lpSum([Y[j]*sizes[j] for j in np.arange(2,6)])
    #Lp_prob += Y[0]*sizes[0] + Y[1]*sizes[1] <= p.lpSum([Y[j]*sizes[j] for j in np.arange(2,6)])
    
    ##############constraint#################
    #first select the  the number of make female in test data
    #then apply the equalized odd constraints assuming 
    #look at all males which have been predicted positve/and all the females predicted negative
    F_test = 0
    M_test = 0
        
    #for i in range(len(y_test)):
    #    if(data1[0][i]==1 and y_test.iloc[i]==1):
    #        M_test= M_test+1
    #    elif(data1[1][i]==1 and y_test.iloc[i]==1):
    #        F_test= F_test+1
    test_count = np.zeros(m, dtype = int)
    '''
    for i in range(len(y_test)):
      for j in range(m): 
        if(data1[j][i]==1 and y_test_pred[i]==1):
            test_count[j] = test_count[j] +1
    '''            
    
    #Lp_prob += (p.lpSum([(X[j])*(data1[0][j])*y_test_pred[j] for j in range(n) if y_test_pred[j]==1])/M_test) <= (p.lpSum([(X[j])*(data1[1][j])*y_test_pred[j] for j in range(n) if y_test_pred[j]==1])/F_test) + 0.0009
    #Lp_prob += (p.lpSum([(X[j])*(data1[0][j])*(1-y_test_pred[j]) for j in range(n) if y_test_pred[j]==0])/(sizes[0]-M_test)) <= (p.lpSum([(X[j])*(data1[1][j])*(1-y_test_pred[j]) for j in range(n) if y_test_pred[j]==0])/(sizes[1]-F_test))+ 0.0009
    
    '''
    for i in range(m):   #TPR constraints
      Lp_prob += (1/test_count[i])*p.lpSum([(X[j])*(data1[i][j])*y_test_pred[j] for j in range(n) if (y_test_pred[j]==1) ]) >= tpr 
      Lp_prob += (1/test_count[i])*p.lpSum([(X[j])*(data1[i][j])*y_test_pred[j] for j in range(n) if (y_test_pred[j]==1 )]) <= tpr + 0.01
    for i in range(m):    #FPR constraints
      Lp_prob += (1/(sizes[i]-test_count[i]))*p.lpSum([(X[j])*(data1[i][j])*(1-y_test_pred[j]) for j in range(n) if (y_test_pred[j]==0)]) >= fpr
      Lp_prob += (1/(sizes[i]-test_count[i]))*p.lpSum([(X[j])*(data1[i][j])*(1-y_test_pred[j]) for j in range(n) if (y_test_pred[j]==0)]) <= fpr + 0.01
    '''
    #Lp_prob += F_test*p.lpSum([(X[j])*(data1[0][j])*y_test_pred[j] for j in range(n) if (y_test_pred[j]==1) ]) <= M_test*p.lpSum([(X[j])*(data1[1][j])*y_test_pred[j] for j in range(n) if (y_test_pred[j]==1 )]) + 0.004
    #Lp_prob += (sizes[1]-F_test)*p.lpSum([(X[j])*(data1[0][j])*(1-y_test_pred[j]) for j in range(n) if (y_test_pred[j]==0)]) <= (sizes[0]-M_test)*p.lpSum([(X[j])*(data1[1][j])*(1-y_test_pred[j]) for j in range(n) if (y_test_pred[j]==0 )]) + 0.004
    


    #Lp_prob += F_test*p.lpSum([(X[j])*(data1[0][j])*y_test_pred[j] for j in range(n) if (y_test_pred[j]==1 and y_test.iloc[j]==1) ]) <= M_test*p.lpSum([(X[j])*(data1[1][j])*y_test_pred[j] for j in range(n) if (y_test_pred[j]==1 and y_test.iloc[j]==1)]) + 0.01
    #Lp_prob += (sizes[1]-F_test)*p.lpSum([(X[j])*(data1[0][j])*(1-y_test_pred[j]) for j in range(n) if (y_test_pred[j]==0 and y_test.iloc[j]==0)]) <= (sizes[0]-M_test)*p.lpSum([(X[j])*(data1[1][j])*(1-y_test_pred[j]) for j in range(n) if (y_test_pred[j]==0 and y_test.iloc[j]==0)]) + 0.01
    
    #Lp_prob += p.lpSum([(X[j])*(data1[0][j])*y_test.iloc[j] for j in range(n) if y_test.iloc[j]==1])/M_test <= p.lpSum([(X[j])*(data1[1][j])*y_test.iloc[j] for j in range(n) if y_test.iloc[j]==1])/F_test + 0.004
    #Lp_prob += (sizes[1]-F_test)*p.lpSum([(X[j])*(data1[0][j])*(1-y_test.iloc[j]) for j in range(n) if y_test.iloc[j]==0])/M_test <= (sizes[0]-M_test)*p.lpSum([(X[j])*(data1[1][j])*(1-y_test.iloc[j]) for j in range(n) if y_test.iloc[j]==0])/F_test + 0.004
    
    for i in range(m):
      #  if i<m:
            Lp_prob += p.lpSum([(X[j])*(data1[i][j]) for j in range(n)]) >= Y[i]*sizes[i]
            Lp_prob += p.lpSum([(X[j])*(data1[i][j]) for j in range(n)]) <= (Y[i]+epsilon)*sizes[i]
    
    beta_avg = 0.5
    alpha = 0
    for i in range(m):
        if beta_actual[i] >= beta_avg:
            Lp_prob += Y[i] >= (1-alpha)*beta_actual[i] + alpha*beta_avg
            Lp_prob += Y[i] <= beta_actual[i]
        else:
            Lp_prob += Y[i] >= (1-alpha)*beta_actual[i] + alpha*beta_avg
            Lp_prob += Y[i] <= beta_avg 
    
           
    #Lp_prob+= p.lpSum([(X[j])*cost[j] for j in range(n)])>=100
        
    #####################################
    #solver = p.CPLEX_PY()
    #solver.buildSolverModel(Lp_prob)
    #Lp_prob.solverModel.parameters.timelimit.set(60)
    #solver.callSolver(P)
    #status = solver.findSolutionValues(Lp_prob)
    #################################################################
    status = Lp_prob.solve(solver)   # Solver 
    print(p.LpStatus[status]) 
    print("objective is:")        
    print(p.value(Lp_prob.objective))
    print("discripency is:") 
    print(p.value(X[n]))
    x=np.zeros(n,dtype=float)

   # The solution status 
    Synth1={}
    Synth2={}
    # # Printing the final solution
    ## compute accuracy, new DP and old DP and NDCG
    count1 = 0
    count2 = 0
    count3 = 0

    fair_scores = []
    for i in range(n):
      fair_scores.append(p.value(X[i]))
      if fair_scores[i]==1.0: count1 = count1 + 1
      elif fair_scores[i]==0.0: count2 = count2 + 1
      else: count3 = count3 + 1

    print(count1,count2,count3)
    #print(np.array(fair_scores))
    print(data['score'].to_numpy())
    
    gr = [ [] for _ in range(m) ]
    grf= [ [] for _ in range(m) ]
      
    #count = np.zeros(r, dtype=int)
      
    DP_list = []
    DPf_list = []
    
    for i in range(n):
          for j in range(2):
            for k in range(2):
              if data.iloc[i,1] == j and data.iloc[i,4] == k:
                gr[2*j+k].append(data.iloc[i,6])
                grf[2*j+k].append(fair_scores[i])
                        
          for j in range(6):
            for k in range(6):
              if data.iloc[i,2]==j and data.iloc[i,5]==k :
                gr[6*j+k+4].append(data.iloc[i,6])   #begins at 4
                grf[6*j+k+4].append(fair_scores[i])

    for i in range(m):
      if sizes[i] != 0:
        DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
        DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

    print(max(DP_list), min(DP_list))
    print(max(DPf_list), min(DPf_list))
    DP = max(DP_list)-min(DP_list)
    DPf = max(DPf_list)-min(DPf_list)
    print("Demogrphic Parity is without/with fair", DP, DPf)
    
    M = 500
    k = 10
    accum_ndcg = 0
    node_cnt = 0
    accum_ndcg_u = 0
    node_cnt_u = 0    
    adj_mtx_fair = lil_matrix((M,M))
    adj_mtx_unfair = lil_matrix((M,M))
    selected_pairs = lil_matrix((M,M))

    for i in range(n):
        adj_mtx_unfair[int(data.iloc[i,0]),int(data.iloc[i,3])] = data.iloc[i,6]
        selected_pairs[int(data.iloc[i,0]),int(data.iloc[i,3])] = 1
        adj_mtx_fair[int(data.iloc[i,0]),int(data.iloc[i,3])] = fair_scores[i]

    #print(adj_mtx_fair)
    #print(np.count_nonzero(adj_mtx_fair))
    print('Utility evaluation (link prediction)')
    s = random.sample(range(M),10000)
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
 
        #pred_edges_fair.append(adj_mtx_fair[node_id][i])
        #pred_edges_unfair.append(adj_mtx_fair[node_id][i])

        #pos_nodes = np.where(node_edges>0)[0]
        #num_pos = len(pos_nodes)
        #num_test_pos = int(len(pos_nodes) / 10) + 1
        #test_pos_nodes = pos_nodes[:num_test_pos]
        #num_pos = len(test_pos_nodes)
        #print(num_pos)

        #if num_pos == 0 or num_pos >= 100:
        #    continue
        #neg_nodes = np.random.choice(np.where(node_edges == 0)[0], 100-num_pos, replace=False)
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
                 if adj_mtx[node_id,i]>0:
                     pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                 else:
                     pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])

        #print(pred_edges_fair_pos)
        pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
        pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
      
        #print(len(pred_edges_unfair))
        #print(len(pred_edges_fair))

        
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

    print('-- ndcg of link prediction for LP score:{}'.format(score))
    print('-- ndcg of link prediction for unfair score:{}'.format(score_u))



def QPG(data,adj_mtx, beta_avg):     #here we are taking two sensitive attrib # make the data here  
    m = 40 # total 40 groups are there 
    n = len(data)
    data1 = np.zeros((m,n))
    sizes_edges = np.zeros(m)
    b = sigmoid(data.iloc[:,6].to_numpy())

    adamic_adar = np.zeros(n, dtype=float)

    M = adj_mtx.shape[0]
    nbr = np.zeros(M, dtype=int)

    G = nx.from_numpy_matrix(adj_mtx)


    for i in range(M):
      for j in range(M):
        if adj_mtx[i][j] > 0:
          nbr[i] = nbr[i] + 1

    for i in range(n):
      u = int(data.iloc[i,0])
      v = int(data.iloc[i,3])
      for j in list(set(G[u]) & set(G[v])):
        adamic_adar[i] = adamic_adar[i] + 1/math.log2(nbr[j])

    a_adar_gr = np.zeros(m, dtype=float)

    sizes=np.zeros(m,dtype=int)
#     report_index(index,data1,e):  


    for i in range(len(data)):
        for j in range(2):
          for k in range(2):
            if data.iloc[i,1]==j and data.iloc[i,4]==k :
              data1[2*j+k][i] = 1
              a_adar_gr[2*j+k] =  a_adar_gr[2*j+k] + adamic_adar[i]
            if data.iloc[i,1]==j and data.iloc[i,4]==k and adj_mtx[int(data.iloc[i,0])][int(data.iloc[i,3])] > 0:
              sizes_edges[2*j+k] = sizes_edges[2*j+k]+1
             
        for j in range(6):
          for k in range(6):
            if data.iloc[i,2]==j and data.iloc[i,5]==k :
              data1[6*j+k+4][i] = 1   #begins at 4
              a_adar_gr[6*j+k+4] =  a_adar_gr[6*j+k+4] + adamic_adar[i]
            if data.iloc[i,2]==j and data.iloc[i,5]==k and adj_mtx[int(data.iloc[i,0])][int(data.iloc[i,3])] > 0:
              sizes_edges[6*j+k+4] = sizes_edges[6*j+k+4]+1



    max_size=0
    for i in range(m):
        count=0
        for j in range(n):
            if data1[i][j]==1:
                count=count+1 
        if count>max_size:
            max_size=count
        sizes[i]=count
    print(sizes)    
    print(sizes_edges/sizes)
    edge_density = np.zeros(m)
    for i in range(m):
      if sizes[i]==0:
        edge_density[i] = 0
      else:
        edge_density[i] = sizes_edges[i]*400000/sizes[i]

    print("data is")
    print(data1)
    print(m,n)

    group = np.zeros(n)
    for i in range(n):
      for j in range(2):
        for k in range(2):
          if data.iloc[i,1] == j and data.iloc[i,4] == k:
            group[i] = group[i] + edge_density[2*j+k]
     
      for j in range(6):
        for k in range(6):
          if data.iloc[i,2]==j and data.iloc[i,5]==k :
            group[i] = group[i] + edge_density[6*j+k+4]

    beta_all = [0.05,0.1,0.2,0.4,0.8]
    for beta in beta_all:
      print("--------------This is for epsilon=",beta,"---------------------------")
      gr = [ [] for _ in range(m) ]

      #count = np.zeros(r, dtype=int)

      random_beta = np.random.rand(m)

      DP_list = []
    
      for i in range(n):
            for j in range(2):
              for k in range(2):
                if data.iloc[i,1] == j and data.iloc[i,4] == k:
                  gr[2*j+k].append(data.iloc[i,6])
                        
            for j in range(6):
              for k in range(6):
                if data.iloc[i,2]==j and data.iloc[i,5]==k :
                  gr[6*j+k+4].append(data.iloc[i,6])   #begins at 4

      for i in range(m):
        if sizes[i] != 0:
          DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
        else:
          DP_list.append(0)
    
        
      
      b = sigmoid(data.iloc[:,6].to_numpy())
      
      '''
      def loss(z):
        return sum((z-b)**2.0)
      bnds = []
      upper = []
      lower = []
      for i in range(m):
        lower.append(DP_list[i]*sizes[i])
        upper.append((DP_list[i]+0.05)*sizes[i])
        bnds.append((0,1))

      linear_constraint = LinearConstraint(data1, lower, upper)
      x0 = np.ones(n)
      res = minimize(loss, x0, method='trust-constr', bounds=bnds, constraints = linear_constraint,options={'verbose': 1})
      print(res.x)
      print("Scipy optimization done")
      '''

      # Define and solve the CVXPY problem.
      Dem = np.array(DP_list)
      Beta = beta*np.ones(m)
      base = 0.02
      x = cp.Variable(n)
      y = cp.Variable(m)
      data_np = data1
      constraints = []
      cost = cp.sum_squares(x - b)
      #cost = cp.sum(cp.kl_div(x,b))
      constraints += [data1 @ x <= (base + beta)*sizes]
      constraints += [data1 @ x >= base*sizes]
      constraints += [x>=0, x<=1]
      #print(constraints)
      #print(cost)
      prob = cp.Problem(cp.Minimize(cost), constraints)
      prob.solve(solver=cp.SCS)

      # Print result.
      print("\nThe optimal value is", prob.value)
      print("The optimal x is")
      print(x.value)
      print("CVXPY optimization done")
      fair_scores = x.value

      #fair_scores = res.x
    
      #print(sizes_edges)
      #print(a_adar_gr/sizes_edges)
      #print(sizes_edges/sizes)


      # # Printing the final solution
      ## compute accuracy, new DP and old DP and NDCG
      count1 = 0
      count2 = 0
      count3 = 0

      #fair_scores = []
      for i in range(n):
        #fair_scores.append(p.value(X[i]))
        if fair_scores[i]==1.0: count1 = count1 + 1
        elif fair_scores[i]==0.0: count2 = count2 + 1
        else: count3 = count3 + 1

      print(count1,count2,count3)
      #print(np.array(fair_scores))
      print(data['score'].to_numpy())
    
      gr = [ [] for _ in range(m) ]
      grf= [ [] for _ in range(m) ]
      
      #count = np.zeros(r, dtype=int)
      
      DP_list = []
      DPf_list = []
    
      for i in range(n):
            for j in range(2):
              for k in range(2):
                if data.iloc[i,1] == j and data.iloc[i,4] == k:
                  gr[2*j+k].append(data.iloc[i,6])
                  grf[2*j+k].append(fair_scores[i])
                        
            for j in range(6):
              for k in range(6):
                if data.iloc[i,2]==j and data.iloc[i,5]==k :
                  gr[6*j+k+4].append(data.iloc[i,6])   #begins at 4
                  grf[6*j+k+4].append(fair_scores[i])

      for i in range(m):
        if sizes[i] != 0:
          DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
          DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

      print(max(DP_list), min(DP_list))
      print(max(DPf_list), min(DPf_list))
      DP = max(DP_list)-min(DP_list)
      DPf = max(DPf_list)-min(DPf_list)
      print("Demogrphic Parity is without/with fair", DP, DPf)
    
      M = 67796
      k = 10
      accum_ndcg = 0
      node_cnt = 0
      accum_ndcg_u = 0
      node_cnt_u = 0    
      adj_mtx_fair = np.zeros((M,M))
      adj_mtx_unfair = np.zeros((M,M))
      selected_pairs = np.zeros((M,M))

      for i in range(n):
          adj_mtx_unfair[int(data.iloc[i,0])][int(data.iloc[i,3])] = data.iloc[i,6]
          selected_pairs[int(data.iloc[i,0])][int(data.iloc[i,3])] = 1
          adj_mtx_fair[int(data.iloc[i,0])][int(data.iloc[i,3])] = fair_scores[i]

      #print(adj_mtx_fair)
      #print(np.count_nonzero(adj_mtx_fair))
      print('Utility evaluation (link prediction)')
      s = random.sample(range(M),10000)
      for node_id in s:
          node_edges = adj_mtx[node_id]
          test_pos_nodes = []
          neg_nodes = []
          for i in range(M):
              if selected_pairs[node_id][i]==1:
                   if adj_mtx[node_id][i]>0:
                       test_pos_nodes.append(i)
                   else:
                       neg_nodes.append(i)
 
        #pred_edges_fair.append(adj_mtx_fair[node_id][i])
        #pred_edges_unfair.append(adj_mtx_fair[node_id][i])

        #pos_nodes = np.where(node_edges>0)[0]
        #num_pos = len(pos_nodes)
        #num_test_pos = int(len(pos_nodes) / 10) + 1
        #test_pos_nodes = pos_nodes[:num_test_pos]
        #num_pos = len(test_pos_nodes)
        #print(num_pos)

        #if num_pos == 0 or num_pos >= 100:
        #    continue
        #neg_nodes = np.random.choice(np.where(node_edges == 0)[0], 100-num_pos, replace=False)
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
              if selected_pairs[node_id][i]==1:
                   if adj_mtx[node_id][i]>0:
                       pred_edges_fair_pos.append(adj_mtx_fair[node_id][i])
                       pred_edges_unfair_pos.append(adj_mtx_unfair[node_id][i])
                   else:
                       pred_edges_fair_neg.append(adj_mtx_fair[node_id][i])
                       pred_edges_unfair_neg.append(adj_mtx_unfair[node_id][i])

        #print(pred_edges_fair_pos)
          pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
          pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
      
        #print(len(pred_edges_unfair))
        #print(len(pred_edges_fair))

        
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

      print('-- ndcg of link prediction for QP score:{}'.format(score))
      print('-- ndcg of link prediction for unfair score:{}'.format(score_u))

def QPG_AA(data,adj_mtx, beta_avg):     #here we are taking two sensitive attrib # make the data here  
    m = 125 # total 40 groups are there 
    n = len(data)
    data1 = np.zeros((m,n))
    sizes_edges = np.zeros(m)

    adamic_adar = np.zeros(n, dtype=float)

    M = adj_mtx.shape[0]
    nbr = np.zeros(M, dtype=int)
    
    #G = nx.from_numpy_matrix(adj_mtx)
   

    for i in range(M):
      for j in range(M):
        if adj_mtx[i,j] > 0:
          nbr[i] = nbr[i] + 1

    '''
    for i in range(n):
      u = int(data.iloc[i,0])
      v = int(data.iloc[i,3])
      for j in list(set(G[u]) & set(G[v])):
        adamic_adar[i] = adamic_adar[i] + 1/math.log2(nbr[j])
    '''
    a_adar_gr = np.zeros(m, dtype=float)

    sizes=np.zeros(m,dtype=int)
#     report_index(index,data1,e):  


    for i in range(len(data)):
        for j in range(2):
          for k in range(2):
            if data.iloc[i,1]==j and data.iloc[i,4]==k :
              data1[2*j+k][i] = 1
              a_adar_gr[2*j+k] =  a_adar_gr[2*j+k] + adamic_adar[i]
            if data.iloc[i,1]==j and data.iloc[i,4]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,3])] > 0:
              sizes_edges[2*j+k] = sizes_edges[2*j+k]+1
             
        for j in range(11):
          for k in range(11):
            if data.iloc[i,2]==j and data.iloc[i,5]==k :
              data1[11*j+k+4][i] = 1   #begins at 4
              a_adar_gr[11*j+k+4] =  a_adar_gr[11*j+k+4] + adamic_adar[i]
            if data.iloc[i,2]==j and data.iloc[i,5]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,3])] > 0:
              sizes_edges[11*j+k+4] = sizes_edges[11*j+k+4]+1



    max_size=0
    for i in range(m):
        count=0
        for j in range(n):
            if data1[i][j]==1:
                count=count+1 
        if count>max_size:
            max_size=count
        sizes[i]=count
    print(sizes)    
    print(sizes_edges/sizes)
    edge_density = np.zeros(m)
    for i in range(m):
      if sizes[i]==0:
        edge_density[i] = 0
      else:
        edge_density[i] = sizes_edges[i]*400000/sizes[i]

    print("data is")
    print(data1)
    print(m,n)

    group = np.zeros(n)
    for i in range(n):
      for j in range(2):
        for k in range(2):
          if data.iloc[i,1] == j and data.iloc[i,4] == k:
            group[i] = group[i] + edge_density[2*j+k]
     
      for j in range(11):
        for k in range(11):
          if data.iloc[i,2]==j and data.iloc[i,5]==k :
            group[i] = group[i] + edge_density[6*j+k+4]

    beta_all = [0.05,0.1,0.2,0.27,0.4,0.8]
    indicator_all = [0] 
    for beta in beta_all:
      print("--------------This is for epsilon=",beta,"---------------------------")
      for indicator in indicator_all:
        print("--------------This is for indicator=",indicator,"---------------------------")
        gr = [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

        random_beta = np.random.rand(m)

        DP_list = []
      
        for i in range(n):
              for j in range(2):
                for k in range(2):
                  if data.iloc[i,1] == j and data.iloc[i,4] == k:
                    gr[2*j+k].append(data.iloc[i,6])
                          
              for j in range(11):
                for k in range(11):
                  if data.iloc[i,2]==j and data.iloc[i,5]==k :
                    gr[11*j+k+4].append(data.iloc[i,6])   #begins at 4

        for i in range(m):
          if sizes[i] != 0:
            DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
          else:
            DP_list.append(0)
      
          
        
        b = sigmoid(data.iloc[:,6].to_numpy())
        
        '''
        def loss(z):
          return sum((z-b)**2.0)
        bnds = []
        upper = []
        lower = []
        for i in range(m):
          lower.append(DP_list[i]*sizes[i])
          upper.append((DP_list[i]+0.05)*sizes[i])
          bnds.append((0,1))

        linear_constraint = LinearConstraint(data1, lower, upper)
        x0 = np.ones(n)
        res = minimize(loss, x0, method='trust-constr', bounds=bnds, constraints = linear_constraint,options={'verbose': 1})
        print(res.x)
        print("Scipy optimization done")
        '''

        # Define and solve the CVXPY problem.
        Dem = np.array(DP_list)
        Beta = beta*np.ones(m)
        base = 0.2
        x = cp.Variable(n)
        y = cp.Variable(m)
        data_np = data1
        constraints = []
        cost1 = cp.sum_squares(cp.multiply(group,(x - b)))
        cost2 = cp.sum_squares(x - b)
        cost3 = cp.sum(cp.kl_div(x,b))
        constraints += [data1 @ x <= (base + beta)*sizes]
        constraints += [data1 @ x >= base*sizes]
        constraints += [x>=0, x<=1]
        #print(constraints)
        #print(cost)
        if indicator==0: #LP
          prob = cp.Problem(cp.Minimize(cost2), constraints)
          prob.solve(solver=cp.SCS)
        elif indicator==1: #QP 
          prob = cp.Problem(cp.Minimize(cost1), constraints)
          prob.solve(solver=cp.SCS)
        else: # KL
          prob = cp.Problem(cp.Minimize(cost3), constraints)
          prob.solve(solver=cp.SCS)        
        # Print result.
        print("\nThe optimal value is", prob.value)
        print("The optimal x is")
        print(x.value)
        print("CVXPY optimization done")
        fair_scores = x.value

        #fair_scores = res.x
      
        #print(sizes_edges)
        #print(a_adar_gr/sizes_edges)
        #print(sizes_edges/sizes)


        # # Printing the final solution
        ## compute accuracy, new DP and old DP and NDCG
        count1 = 0
        count2 = 0
        count3 = 0

        #fair_scores = []
        for i in range(n):
          #fair_scores.append(p.value(X[i]))
          if fair_scores[i]==1.0: count1 = count1 + 1
          elif fair_scores[i]==0.0: count2 = count2 + 1
          else: count3 = count3 + 1

        print(count1,count2,count3)
        #print(np.array(fair_scores))
        print(data['score'].to_numpy())
      
        gr = [ [] for _ in range(m) ]
        grf= [ [] for _ in range(m) ]
        
        #count = np.zeros(r, dtype=int)
        
        DP_list = []
        DPf_list = []
      
        for i in range(n):
              for j in range(2):
                for k in range(2):
                  if data.iloc[i,1] == j and data.iloc[i,4] == k:
                    gr[2*j+k].append(data.iloc[i,6])
                    grf[2*j+k].append(fair_scores[i])
                          
              for j in range(11):
                for k in range(11):
                  if data.iloc[i,2]==j and data.iloc[i,5]==k :
                    gr[11*j+k+4].append(data.iloc[i,6])   #begins at 4
                    grf[11*j+k+4].append(fair_scores[i])

        for i in range(m):
          if sizes[i] != 0:
            DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
            DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

        print(max(DP_list), min(DP_list))
        print(max(DPf_list), min(DPf_list))
        DP = max(DP_list)-min(DP_list)
        DPf = max(DPf_list)-min(DPf_list)
        print("Demogrphic Parity is without/with fair", DP, DPf)
      
        M = 450
        k = 10
        accum_ndcg = 0
        node_cnt = 0
        accum_ndcg_u = 0
        node_cnt_u = 0    
        adj_mtx_fair = np.zeros((M,M))
        adj_mtx_unfair = np.zeros((M,M))
        selected_pairs = np.zeros((M,M))

        for i in range(n):
            adj_mtx_unfair[int(data.iloc[i,0])][int(data.iloc[i,3])] = data.iloc[i,6]
            selected_pairs[int(data.iloc[i,0])][int(data.iloc[i,3])] = 1
            adj_mtx_fair[int(data.iloc[i,0])][int(data.iloc[i,3])] = fair_scores[i]

        #print(adj_mtx_fair)
        #print(np.count_nonzero(adj_mtx_fair))
        print('Utility evaluation (link prediction)')
        s = random.sample(range(M),M)
        counter = 0
        counter1 = 0
        for node_id in s:
            node_edges = adj_mtx[node_id]
            test_pos_nodes = []
            neg_nodes = []
            for i in range(M):
                if selected_pairs[node_id][i]==1:
                    if adj_mtx[node_id,i]>0:
                        test_pos_nodes.append(i)
                    else:
                        neg_nodes.append(i)
  
          #pred_edges_fair.append(adj_mtx_fair[node_id][i])
          #pred_edges_unfair.append(adj_mtx_fair[node_id][i])

          #pos_nodes = np.where(node_edges>0)[0]
          #num_pos = len(pos_nodes)
          #num_test_pos = int(len(pos_nodes) / 10) + 1
          #test_pos_nodes = pos_nodes[:num_test_pos]
          #num_pos = len(test_pos_nodes)
          #print(num_pos)

          #if num_pos == 0 or num_pos >= 100:
          #    continue
          #neg_nodes = np.random.choice(np.where(node_edges == 0)[0], 100-num_pos, replace=False)
            num_pos = len(test_pos_nodes)
            all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes)) 
            all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


            all_eval_edges[:num_pos] = 1
          #print(all_eval_edges)

        
          #in pred_edges all positive edges should be before and then negative edge sores
            edges = []
            pred_edges_fair_pos = []
            pred_edges_fair_neg = []

            pred_edges_unfair_pos = []
            pred_edges_unfair_neg = []

            for i in range(M):
                if selected_pairs[node_id][i]==1:
                    if adj_mtx[node_id,i]>0:
                        pred_edges_fair_pos.append(adj_mtx_fair[node_id][i])
                        pred_edges_unfair_pos.append(adj_mtx_unfair[node_id][i])
                    else:
                        pred_edges_fair_neg.append(adj_mtx_fair[node_id][i])
                        pred_edges_unfair_neg.append(adj_mtx_unfair[node_id][i])

          #print(pred_edges_fair_pos)
            pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
            pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
        
          #print(len(pred_edges_unfair))
          #print(len(pred_edges_fair))

            
            if len(pred_edges_unfair) >=k:
              #pred_edges_unfair = np.array(pred_edges_unfair)
                rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                ranked_node_edges = all_eval_edges[rank_pred_keys]
                ndcg_u = ndcg_at_k(ranked_node_edges, k)
                
                if ndcg_u != 0.0:
                    print("Top edges unfair are", ranked_node_edges[:10])
                    accum_ndcg_u += ndcg_u
                    print(ndcg_u, node_cnt_u)
                    node_cnt_u += 1
  
            if len(pred_edges_fair) >=k:
              #pred_edges_fair = np.array(pred_edges_fair)
                rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                ranked_node_edges = all_eval_edges[rank_pred_keys]
                ndcg = ndcg_at_k(ranked_node_edges, k)
                if ndcg != 0.0:
                    print("Top edges fair are", ranked_node_edges[:10])
                    accum_ndcg += ndcg
                    print(ndcg, node_cnt)
                    node_cnt += 1
            if ndcg_u > ndcg: counter = counter+1
            if ndcg > ndcg_u: counter1 = counter1+1
        score = accum_ndcg/node_cnt
        score_u = accum_ndcg_u/node_cnt_u
        print("unfair scores are better in", counter)
        print("fair scores are better in", counter1)
        print('-- ndcg of link prediction for QP score:{}'.format(score))
        print('-- ndcg of link prediction for unfair score:{}'.format(score_u))



M = 450
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
adj_mtx = lil_matrix((M,M))
# load the adjacency matrix of size MxM of training or testing set

print('loading data ...')
with open('nba_edge.csv', 'r') as fin:
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


data = pd.read_csv('allNBA_country_age.csv', sep=',')
b = sigmoid(data.iloc[:,6].to_numpy())
epsilon = [0.001,0.01,0.02,0.1,0.2,0.3,0.4]
#for i in epsilon:
#  print("------------------This is for beta=",i,"------------------------")
  #LPG(data,adj_mtx,i)

QPG_AA(data,adj_mtx,0.5)
#    LPG(data,adj_mtx,i)
