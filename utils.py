import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import *
from time import perf_counter
import math
import scipy.sparse as sp
import random
import torch.nn as nn
import torch.multiprocessing as mp
import multiprocessing
from collections import defaultdict
from ctypes import *
import dgl
import os

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

import math
def exponential_decay(initial_lr, decay_rate, current_epoch):
    return initial_lr * math.exp(-decay_rate * current_epoch)

def preprocess_citation(adj, features, normalization="FirstOrderGCN", r=0.5):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj, r)
    features = row_normalize(features)
    return adj, features

def preprocess_citation_multi(adj, features, num_hops, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization_multi(normalization)
    adj = adj_normalizer(adj, num_hops)
    features = row_normalize(features)
    return adj, features

def PPR(A, pr):
    pagerank_prob=pr  #0.85 -1   0.65best 68
    pr_prob = 1 - pagerank_prob
    A_hat   = A + torch.eye(A.size(0)).to(A.device)
    D       = torch.diag(torch.sum(A_hat,1))
    D       = D.inverse().sqrt()
    A_hat   = torch.mm(torch.mm(D, A_hat), D)
    Pi = pr_prob * ((torch.eye(A.size(0)).to(A.device) - (1 - pr_prob) * A_hat).inverse())
    Pi = Pi.cpu()
    return Pi

def calc_ppr(adj_dict, pr, device):
    # ppr={}
    adj = adj_dict.clone()
    # adj.storage._value = None
    adj = adj.to_dense()
    adj_T = adj.t()
    A = torch.zeros(adj.shape[0], adj.shape[0]).to(device)
    A = adj
    PPRM=PPR(A, pr)
    ppr=(PPRM)
    ###待优化
    return ppr


def random_walk_sparse(neighbors, num_steps):
    num_nodes = len(neighbors)
    adj_matrix_rw_buffer = []
    # neighbors = [adj_matrix[node].coalesce().indices().squeeze(0).tolist() for node in range(num_nodes)]
    for step in range(1, num_steps+1):
        adj_matrix_rw = torch.sparse.FloatTensor(num_nodes, num_nodes)
        visited = [set() for _ in range(num_nodes)]
        for node in range(num_nodes):
            # print(node, num_nodes)
            current_node = node
            flag = True
            for current_step in range(step):
                current_neighbors = neighbors[current_node]
                valid_neighbors = [n for n in current_neighbors if n not in visited[node]]
                valid_neighbors = current_neighbors
                # print(valid_neighbors)
                if not valid_neighbors:
                    flag = False
                    break
                # print("111")
                # print(node)
                next_node_idx = random.choice(range(len(valid_neighbors)))
                current_node = valid_neighbors[next_node_idx]
            if flag:
                adj_matrix_rw.add_(torch.sparse.FloatTensor(torch.LongTensor([[node], [current_node]]), torch.FloatTensor([1.0]), torch.Size([len(neighbors), len(neighbors)])))
        adj_matrix_rw_buffer.append(adj_matrix_rw)
    return adj_matrix_rw_buffer

###优化代码:
# torch.multiprocessing    8.1未实现，存在bug
#Use a more efficient data structure for the  visited  set: Instead of using a Python set, you can use a boolean array to keep track of visited nodes.
# ###每个点步数相互独立###
def random_walk(adj_matrix, weight_score, num_steps):
    num_nodes = adj_matrix.size(0)
    adj_matrix_rw_buffer = []
    neighbors = [adj_matrix[node].nonzero().flatten().tolist() for node in range(num_nodes)]
    for step in range(1, num_steps+1):
        adj_matrix_rw = torch.zeros_like(adj_matrix)
        visited = [set() for _ in range(num_nodes)]
        for node in range(num_nodes):
            current_node = node
            # visited[node].add(current_node)  ###注释掉是考虑自环的可能,（不注释掉acc下降很厉害，待进一步验证）,这一行无法实现目标
            flag = True
            for current_step in range(step):
                current_neighbors = neighbors[current_node]
                valid_neighbors = [n for n in current_neighbors if n not in visited[node]]
                if not valid_neighbors:
                    flag = False
                    break
                # next_node_idx = random.choice(range(len(valid_neighbors)))                
                next_node_idx = random.choices(range(len(valid_neighbors)), weights=weight_score[node][valid_neighbors])
                current_node = valid_neighbors[next_node_idx[0]]  ###valid_neighbors才对啊
                # visited[node].update(current_neighbors) #visited[node].update(current_neighbors)  # 无跳回实现 Add all current hop nodes to visited, 使用这一行可以达到目标
                #但是这一行似乎实现过拟合了，我们只是要访问过的节点不被访问就可以了，没必要让所有邻居都不被访问到
                # visited[node].update([current_node])
            if flag:
                adj_matrix_rw[node, current_node] = 1 ###只记录终点的位置，途径点都不记录  也就是说不做递归
        adj_matrix_rw = adj_matrix_rw.to_sparse().coalesce()
        adj_matrix_rw_buffer.append(adj_matrix_rw)
    return adj_matrix_rw_buffer

# ### 每个点的步数相互依赖###
# import random
# def random_walk(adj_matrix, num_steps):
#     num_nodes = adj_matrix.size(0)
#     adj_matrix_rw_buffer = []
#     neighbors = [adj_matrix[node].nonzero().flatten().tolist() for node in range(num_nodes)]
#     visited = [set() for _ in range(num_nodes)] 
#     for node in range(num_nodes):
#         walk = [node]
#         ###visited[node].add(node)  ###注释掉是考虑自环的可能,不注释掉acc下降很厉害
#         flag = True
#         for current_step in range(1, num_steps+1):
#             current_node = walk[-1]
#             current_neighbors = neighbors[current_node]
#             valid_neighbors = [n for n in current_neighbors if n not in visited[node]]
#             if not valid_neighbors:
#                 flag = False
#                 break
#             next_node_idx = random.choice(range(len(valid_neighbors)))
#             current_node = valid_neighbors[next_node_idx]
#             #visited[node].update(current_neighbors)  # Add all current hop nodes to visited
#             walk.append(current_node)#adj_matrix_rw[node, current_node] = 1
#         adj_matrix_rw_buffer.append(walk)
#     ###根据random walk重构稀疏矩阵###
#     buffer_row, buffer_col = [], []
#     for i in range(1, num_steps+1):
#         row, col = [], []
#         for node, pairs in enumerate(adj_matrix_rw_buffer):
#             if i <= len(pairs)-1:
#                 row.append(node)
#                 col.append(pairs[i])
#         buffer_row.append(row)
#         buffer_col.append(col)
#     concatenated_list = [torch.stack([torch.tensor(buffer_row[i]), torch.tensor(buffer_col[i])]) for i in range(num_steps)]
#     adj_matrix_rw_buffer = []
#     for result in concatenated_list:
#         indices = result
#         values = torch.ones(result.shape[1])
#         shape = torch.Size([num_nodes, num_nodes])
#         adj_matrix_rw_buffer.append(torch.sparse.FloatTensor(indices, values, shape))
#     return adj_matrix_rw_buffer

def index_to_torch_sparse(result):
    row_tensor = torch.tensor(result[0])
    col_tensor = torch.tensor(result[1])
    # Concatenate the tensors
    indices = torch.cat((row_tensor.unsqueeze(0), col_tensor.unsqueeze(0)), dim=0)
    values = torch.ones(len(row_tensor))
    # if (row_tensor[-1] + 1 != len(row_tensor)):
    #     print("assert")
    shape = torch.Size([row_tensor[-1]+ 1,row_tensor[-1] + 1]) #这行实现解决了下面这行的bug
    # shape = torch.Size([len(row_tensor), len(row_tensor)])
    return torch.sparse.FloatTensor(indices, values, shape)

# def preprocess_citation_RW(adj, features, num_steps, num_rws, device, seed, dataset):
#     if dataset!='ogbn-arxiv' and dataset!='ogbn-products':
#         t = perf_counter()
#         adj_eye = adj + sp.eye(adj.shape[0])
#         adj_raw = sparse_mx_to_torch_sparse_tensor(adj_eye)#.to(device)
#     else:
#         t = perf_counter()
#         adj_raw = adj
#     adj_matrix_rw_total = []
#     import dgl
#     g = dgl.graph((adj_raw._indices()[0].tolist(),adj_raw._indices()[1].tolist()))
#     t = perf_counter()
#     nodes = [i for i in range(g.num_nodes())]
#     adj_matrix_rw_total = []
#     for i in range(num_rws):
#         output_buffer = []
#         for hop in range(1, num_steps + 1):
#             result = []
#             a = dgl.sampling.random_walk(g, nodes, length=hop)
#             result.append(nodes)
#             result.append(a[0][:,-1].tolist())
#             output = index_to_torch_sparse(result)
#             output_buffer.append(output)
#         adj_matrix_rw_total.append(output_buffer)
#         # print("finish one round")
#     output= []
#     for i in range(num_steps):  #
#         sum_output = adj_matrix_rw_total[0][i]
#         for j in range(1, num_rws):
#             sum_output += adj_matrix_rw_total[j][i]
#         sum_output = sum_output.coalesce()
#         sum_output = sum_output/num_rws
#         output.append(sum_output)
#     print("pre time", perf_counter()-t)
#     if dataset !='reddit' and dataset !='ogbn-arxiv':
#         print("normalize the features")
#         features = row_normalize(features)
#     return output, features

def preprocess_citation_RW(adj, features, num_steps, num_rws, device, seed, dataset):
    if dataset!='ogbn-arxiv' and dataset!='ogbn-products':
        t = perf_counter()
        adj_eye = adj + sp.eye(adj.shape[0])
        adj_raw = sparse_mx_to_torch_sparse_tensor(adj_eye)#.to(device)
    else:
        t = perf_counter()
        adj_raw = adj
    adj_matrix_rw_total = []
    import ctypes
    cpp_library = ctypes.CDLL('/home/lyx/SGC/rw.so')
    # 定义 C++ 函数的签名
    cpp_library.set_seed(ctypes.c_uint(seed))
    cpp_library.random_walk_interface.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    cpp_library.random_walk_interface.restype = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))
    row_indices = adj_raw.coalesce().indices()[0]
    col_indices = adj_raw.coalesce().indices()[1]
    num_nonzero_per_row = torch.bincount(row_indices).tolist()
    neighbors = torch.split(col_indices, num_nonzero_per_row)
    neighbors_list = [neighbor.tolist() for neighbor in neighbors]
    # matrix_data = adj_raw.to_dense().tolist()
    matrix_ptr = (ctypes.POINTER(ctypes.c_float) * len(neighbors_list))()
    index_ptr = (ctypes.c_int * len(num_nonzero_per_row))(*num_nonzero_per_row)
    for i in range(len(neighbors_list)):
        matrix_ptr[i] = (ctypes.c_float * len(neighbors_list[i]))(*neighbors_list[i])
    ## 调用 C++ 函数并传递矩阵指针
    if dataset != 'ogbn-arxiv':
        for num_rw in range(num_rws):
            output_buffer = []
            adj_matrix_rw = cpp_library.random_walk_interface(matrix_ptr, num_steps, adj_raw.size(0), index_ptr)
            result_buffer = [[[adj_matrix_rw[k][i][j] for j in range(adj_matrix_rw[k][2][0])] for i in range(2)] for k in range(num_steps)] #[steps, 2, index]
            # Convert the indices to tensors
            for result in result_buffer:
                output = index_to_torch_sparse(result)
                output_buffer.append(output)
            adj_matrix_rw_total.append(output_buffer)
            # print("finish 1 round")
        output= []
        # output.append(adj_raw) #加hop=0的矩阵(错误，本身的RW就已经做了hop=1,不存在hop=0这一概念) 
        for i in range(num_steps):  #
            sum_output = adj_matrix_rw_total[0][i]
            for j in range(1, num_rws):
                sum_output += adj_matrix_rw_total[j][i]
            sum_output = sum_output.coalesce()  
            sum_output = sum_output/num_rws
            output.append(sum_output)
    if dataset == 'ogbn-arxiv':
        file = f"/home/lyx/SGC/RW_ogbn/rw_{num_rws}_degree_{num_steps}_seed_{seed}.pt"
        # torch.save(output, file)
        output = torch.load(file)
    time = perf_counter() - t
    print("random walk time: ", time)
    if dataset !='reddit' and dataset !='ogbn-arxiv':
        print("normalize the features")
        features = row_normalize(features)
    return output, features

def preprocess_citation_RW(adj, features, num_steps, num_rws, device, seed, dataset):
    adj_raw = adj + sp.eye(adj.shape[0])
    adj_raw = sparse_mx_to_torch_sparse_tensor(adj_raw).to(device)
    adj_matrix_rw_total = []

    row_indices = adj_raw.coalesce().indices()[0]
    col_indices = adj_raw.coalesce().indices()[1]
    num_nonzero_per_row = torch.bincount(row_indices)
    neighbors = torch.split(col_indices, num_nonzero_per_row.tolist())
    neighbors_list = [neighbor.tolist() for neighbor in neighbors]
    weight_score = calc_ppr(adj_raw, 0.95, device)
    # torch.sum(ppr_sum[key_A], dim = 0)
    for num_rw in range(num_rws):
        adj_matrix_rw = random_walk(adj_raw.to_dense(), weight_score, num_steps)
        # adj_matrix_rw = random_walk_sparse(neighbors_list, num_steps)
        adj_matrix_rw_total.append(adj_matrix_rw) #[rws, steps, adj]
        print("finish 1 round")
    output= []
    for i in range(num_steps):  #
        sum_output = 0
        for j in range(num_rws):
            sum_output = sum_output + adj_matrix_rw_total[j][i].to_dense()
        #print(exponential_decay(initial_lr, decay_rate, i))
        ## normalization###
        r = 0
        #print("in here")
        #adj_norm = ((sum/num_rw)).to_sparse().coalesce()
        # sum_output = j.to_sparse().coalesce()
        row_sum = sum_output.to_dense().sum(dim=1)
        d_inv_sqrt = torch.pow(row_sum, r-1).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)  #.to_sparse()
        d_right_sqrt = torch.pow(row_sum, -r).flatten()
        d_right_sqrt[torch.isinf(d_right_sqrt)] = 0.
        d_mat_right_sqrt = torch.diag(d_right_sqrt) #.to_sparse()
        adj_norm = d_mat_inv_sqrt.matmul(sum_output).matmul(d_mat_right_sqrt)
        ### normalization###
        # print(adj_norm._nnz())
        output.append(adj_norm)
    features = row_normalize(features)
    return output, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def group_indices(arr):
    indices = {}
    for i, num in enumerate(arr):
        if num in indices:
            indices[num].append(i)
        else:
            indices[num] = [i]
    return indices

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)   ### AAX     (AX+A2X)/2
    precompute_time = perf_counter()-t
    return features, precompute_time

def sga_precompute(features, adj2_list_final, use_weight):
    t = perf_counter()
    output = 0
    if use_weight!=True:
        for adj in adj2_list_final:
            output = output + torch.spmm(adj, features)
        output = (output)/len(adj2_list_final)
        precompute_time = perf_counter()-t
        return output, precompute_time
    else:
        output = []
        for adj in adj2_list_final:
            output.append(torch.spmm(adj, features))
        precompute_time = perf_counter()-t
        return output, precompute_time
    
def ssgc_precompute(features, adj, degree): ############### acc跟github版本不同的原因，(1-alpha) *  的位置不同
    alpha = 0.05
    t_start = perf_counter()
    adj_now = adj
    emb = alpha * features #+ (features/degree)
    if features.device.type == 'cuda':
        print("Tensor is on GPU")
    else:
        print("Tensor is on CPU")
    for i in range(degree):
        t = perf_counter()
        features = torch.spmm(adj, features)   ### AAX     (AX+A2X)/2
        # torch.save(features, f'/home/lyx/SGC/papers100M_store/walks_5_hop_6/features_{i}.pt')
        # print(adj_now._nnz())
        # adj_now = torch.spmm(adj, adj_now)
        emb = emb + (1-alpha)*features/degree
        precompute_time = perf_counter()-t
        print("precompute time {:.4f}s".format(precompute_time))
    precompute_time = perf_counter()-t_start
    return emb, precompute_time

# ###SSGC的另一种实现
# def sgc_precompute(features, adj, degree, alpha):
#     t = perf_counter()
#     ori_features = features
#     emb = features
#     for i in range(degree):
#         features = (1-alpha) * torch.spmm(adj, features)
#         emb = emb + features
#     emb /= degree
#     emb = emb + alpha * ori_features
#     precompute_time = perf_counter()-t
#     return emb, precompute_time

def ssgc_mask_precompute(features, adj2_list_final, use_weight):
    alpha = 0.05
    degree = len(adj2_list_final)
    # print(degree) 
    t = perf_counter()
    # emb = alpha * features
    emb = 0
    for i, adj in enumerate(adj2_list_final):
        features_now = torch.spmm(adj/20, features)   ### AX AAX AAAX
        print("finish one propagation")
        torch.save(features_now, f'/home/lyx/SGC/papers100M_store/walks_30_hop_5/features_{i}.pt')
        # features_now = 0
        # emb = emb + (1-alpha)*features_now/degree
    precompute_time = perf_counter()-t
    return emb, precompute_time

# def ssgc_mask_precompute(features, adj2_list_final, use_weight):
#     alpha = 0.05
#     degree = 6
#     # print(degree) 
#     t = perf_counter()
#     emb = alpha * features
#     for i in range(degree):
#         # features_now = torch.spmm(adj, features)   ### AX AAX AAAX
#         features_now = torch.load(f'/home/lyx/SGC/papers100M_store/walks_5_hop_6/features_{i}.pt')
#         emb = emb + (1-alpha)*features_now/degree
#     precompute_time = perf_counter()-t
#     return emb, precompute_time

# def ssgc_mask_precompute(features, adj2_list_final, degree):
#     alpha = 0.05
#     t = perf_counter()
#     ori_features = features
#     emb = features
#     degree = len(adj2_list_final)
#     for adj in adj2_list_final:
#         features = (1-alpha) * torch.spmm(adj, features)   ### AAX     (AX+A2X)/2
#         emb = emb + features
#     emb /= degree
#     emb = emb + alpha * ori_features
#     precompute_time = perf_counter()-t
#     return emb, precompute_time
    
# def ssgc_mask_precompute(features, adj2_list_final, use_weight):
#     alpha = 0.05
#     ori_features = features
#     degree = len(adj2_list_final)
#     # print(degree)
#     t = perf_counter()
#     emb = features
#     for i, adj in enumerate(adj2_list_final):
#         features = (1-alpha)*torch.spmm(adj, features)   ### AX AAX AAAX
#         emb = emb + features
#     emb /= degree
#     emb = emb + alpha * ori_features
#     precompute_time = perf_counter()-t
#     return emb, precompute_time


def sign_precompute(features, adj, degree):
    t = perf_counter()
    emb = []
    emb.append(features)
    for i in range(degree):
        features = torch.spmm(adj, features)   ### AAX     (AX+A2X)/2
        emb.append(features)
    precompute_time = perf_counter()-t
    # # ########## test for over-smoothing###########
    # print("test for over-smoothing problem")
    # adj_now = adj
    adj_buffer = []
    # adj_buffer.append(adj_now)
    # for i in range(degree):
    #     adj_now = torch.spmm(adj, adj_now)   ###
    #     adj_buffer.append(adj_now)
    # components = []
    sub_results = []
    # for i, struct in enumerate(adj_buffer):
    #     num_components, labels = sp.csgraph.connected_components(struct.to_dense().cpu().numpy(), directed=False, return_labels=True)
    #     components.append(labels)
    #     print("current hop {:}:, num_components:{:}, nnz:{:}".format(i, num_components, struct._nnz()))
    #     sub_results.append(group_indices(labels))
    # precompute_time = perf_counter()-t
    # ########## test for over-smoothing###########
    return emb, precompute_time, sub_results, adj_buffer

def gbp_precompute(features, adj, degree, alpha):
    t = perf_counter()
    emb = features*alpha
    for i in range(1, degree+1):
        features = torch.spmm(adj, features)   ### AAX     (AX+A2X)/2
        w_dynamic = alpha * math.pow(1-alpha, i)
        emb = emb + w_dynamic * features
    precompute_time = perf_counter()-t
    return emb, precompute_time

def gbp_mask_precompute(features, adj2_list_final, alpha):
    t = perf_counter()
    emb = features*alpha
    for i, adj in enumerate(adj2_list_final):
        features = torch.spmm(adj, features)
        w_dynamic = alpha * math.pow(1-alpha, i+1)
        emb = emb + w_dynamic * features
    precompute_time = perf_counter()-t
    return emb, precompute_time

# def ssgc_mask_precompute(features, adj2_list_final, use_weight):
#     alpha = 0.01
#     degree = len(adj2_list_final)
#     # print(degree)
#     t = perf_counter()
#     emb = alpha*features
#     if use_weight!=True:
#         # for adj in adj2_list_final:
#         #     features = torch.spmm(adj, features)   ### AAX     (AX+A2X)/2
#         #     emb = emb + (1-alpha)*features/degree  #(1-alpha)
#         #     # alpha = alpha*(1+alpha)
#         #     # print(alpha)
#         # precompute_time = perf_counter()-t

#         cur_rate = 1.0
#         beta = 0.
#         for i, adj in enumerate(adj2_list_final):
#             cur_rate = cur_rate -  beta * i
#             features = torch.spmm(adj, features)   ### AX AAX AAAX
#             emb = emb + cur_rate*features/degree  #(1-alpha)
#         precompute_time = perf_counter()-t
#         return emb, precompute_time
#     if use_weight:
#         output = []
#         for adj in adj2_list_final:
#             output.append(torch.spmm(adj, features))
#         precompute_time = perf_counter()-t
#         return output, precompute_time

def sign_mask_precompute(features, adj2_list_final, use_weight):
    emb = []
    sub_results = []
    # features = features.cpu()
    emb.append(features)
    degree = len(adj2_list_final)
    # print(degree)
    t = perf_counter()
    for i, adj in enumerate(adj2_list_final):
        features_now = torch.spmm(adj, features)   ### AX AAX AAAX
        emb.append(features_now)
    precompute_time = perf_counter()-t

    # t = perf_counter()
    # a = torch.stack(adj2_list_final, dim=0)
    # c = torch.bmm(a, features.unsqueeze(0).expand(10, -1, -1))
    # precompute_time = perf_counter()-t

    # for i, struct in enumerate(adj2_list_final):
    #     num_components, labels = sp.csgraph.connected_components(struct.to_dense().cpu().numpy(), directed=False, return_labels=True)
    #     print("current hop {:}:, num_components:{:}, nnz:{:}".format(i+1, num_components, struct._nnz()))
    #     sub_results.append(group_indices(labels))
    return emb, precompute_time, sub_results
# def ssgc_mask_precompute(features, adj2_list_final, use_weight):
#     alpha = 0.05
#     degree = len(adj2_list_final)
#     print(degree)
#     t = perf_counter()
#     ori_features = features
#     emb = features
#     if use_weight!=True:
#         for adj in adj2_list_final:
#             features = torch.spmm(adj, features)   ### AAX     (AX+A2X)/2
#             emb = emb + (1-alpha)*features
#     emb /= degree
#     emb = emb + alpha * ori_features
#     precompute_time = perf_counter()-t
#     return emb, precompute_time

def set_seed(seed, cuda):
    dgl.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    if cuda: torch.cuda.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)    
    # dgl.random.seed(seed)


def RW_citation(dataset_str="cora", normalization="AugNormAdj", num_hops=2, num_wks = 1, cuda=True, model='SGC', device = f"cuda:{1}", seed = 1, r = 0.5):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    
    if model.startswith('RW'):
        adj2_list, features_1 = preprocess_citation_RW(adj, features, num_hops, num_wks, device, seed, dataset_str)
    elif model.startswith('SGA') :
        adj2_list, features_1 = preprocess_citation_multi(adj, features, num_hops, normalization)
    else:
        adj1, features_1 = preprocess_citation(adj, features, normalization, r)
    # porting to pytorch
    features_1 = torch.FloatTensor(np.array(features_1.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    if model.startswith('SGA') or model.startswith('RW'):
        adj2_list_new = []
        for adj2 in adj2_list:
            #adj2 = sparse_mx_to_torch_sparse_tensor(adj2).float()
            adj2_list_new.append(adj2)
    else:
        adj1 = sparse_mx_to_torch_sparse_tensor(adj1).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    t = perf_counter()
    if cuda:
        features_1 = features_1.to(device)
        if model.startswith('SGA') or model.startswith('RW'):
            adj2_list_final = []
            for adj2 in adj2_list_new: 
                adj2 = adj2.to(device)
                adj2_list_final.append(adj2)
        else:
            adj1 = adj1.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
    pre_time = perf_counter() - t
    print("Pre_trans time: {:.4f}s".format(pre_time))
    if model.startswith('SGA') or model.startswith('RW'):
        return adj2_list_final, features_1, labels, idx_train, idx_val, idx_test
    else:
        return adj1, features_1, labels, idx_train, idx_val, idx_test

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(dataset = 'reddit', normalization="AugNormAdj", num_hops = 2,  num_wks = 1, cuda=True, model = 'SGC', device = f"cuda:{0}", seed = 1):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    if model.startswith('RW'):
        adj2_list, _ = preprocess_citation_RW(adj, features, num_hops, num_wks, device, seed, dataset)
        adj2_list_train, _ = preprocess_citation_RW(train_adj, features[train_index], num_hops, num_wks, device, seed, dataset)
    elif model.startswith('SGA') :
        adj2_list, _ = preprocess_citation_multi(train_adj, features, num_hops, normalization)
    else:
        adj1, _ = preprocess_citation(train_adj, features, normalization)
    # porting to pytorch
    labels = torch.LongTensor(labels)
    # if model.startswith('SGA') or model.startswith('RW'):
    #     adj2_list_new = []
    #     for adj2 in adj2_list:
    #         #adj2 = sparse_mx_to_torch_sparse_tensor(adj2).float()
    #         adj2_list_new.append(adj2)
    #     adj2_list_new_train = []
    #     for adj2_train in adj2_list_train:
    #         #adj2 = sparse_mx_to_torch_sparse_tensor(adj2).float()
    #         adj2_list_new_train.append(adj2_train)

    if cuda:
        features = features.to(device)
        if model.startswith('SGA') or model.startswith('RW'):
            adj2_list_final = []
            for adj2 in adj2_list: 
                adj2 = adj2.to(device)
                adj2_list_final.append(adj2)
            adj2_list_final_train = []
            for adj2_train in adj2_list_train: 
                adj2_train = adj2_train.to(device)
                adj2_list_final_train.append(adj2_train)
        else:
            adj1 = adj1.to(device)
        labels = labels.to(device)

    if model.startswith('SGA') or model.startswith('RW'):
        return adj2_list_final, adj2_list_final_train, features, labels, train_index, val_index, test_index
    else:
        return adj1, features, labels, train_index, val_index, test_index


import dgl
import dgl.function as fn
def neighbor_average_features(g, num_hops):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, num_hops + 1):
        g.update_all(
            fn.copy_u(f"feat_{hop-1}", "msg"), fn.mean("msg", f"feat_{hop}")
        )
    res = []
    for hop in range(num_hops + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res

from dataset import load_dataset
def prepare_data(dataset="ogbn-arxiv", normalization="AugNormAdj", num_hops=2, num_wks = 1, cuda=True, model='SGC', device = f"cuda:{2}", seed = 1, r = 0.5, use_dgl = False):

    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_dataset(dataset, device)
    g, adj_raw, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data  #n_classes = labels.max().item()+1
    if model.startswith('RW'):
        adj2_list, features_1 = preprocess_citation_RW(adj_raw, features.numpy(), num_hops, num_wks, device, seed, 'ogbn-arxiv')
    if model.startswith('SGA'):
        g = None
        adj2_list, features_1 = preprocess_citation_multi(adj_raw.to(device), features.to(device), num_hops, 'AugNormAdj')
    if not use_dgl:
        adj1 = adj_raw
        features_1 = features
    else:
        feats = neighbor_average_features(g, num_hops)
    features_1 = torch.FloatTensor(np.array(features_1)).float()

    t = perf_counter()
    adj2_list_final = []
    if cuda:
        features_1 = features_1.to(device)
        if model.startswith('SGA') or model.startswith('RW'):
            for adj2 in adj2_list:
                adj2 = adj2.to(device)
                adj2_list_final.append(adj2)
                adj1 = None
        else:
            adj1 = adj1.to(device)
    # in_feats = g.ndata["feat"].shape[1]
    # feats = neighbor_average_features(g, args)
    pre_time = perf_counter() - t
    print("Pre_trans time: {:.4f}s".format(pre_time))
    labels = labels.to(device)
    # move to device
    train_index = train_nid.to(device)
    val_index = val_nid.to(device)
    test_index = test_nid.to(device)
    return adj2_list_final, adj1, features_1, labels, n_classes, train_index, val_index, test_index
