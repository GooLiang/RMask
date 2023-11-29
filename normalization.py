import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from time import perf_counter
from torch_sparse import SparseTensor

def cosin_similarities(feat):
    norms = torch.norm(feat, dim=1, keepdim=True)
    normalized_matrix = feat / norms
    nan_mask = torch.isnan(normalized_matrix)
    normalized_matrix = torch.where(nan_mask, 0, normalized_matrix)
    cos_similarities = torch.mm(normalized_matrix, normalized_matrix.t())
    cos_similarities.fill_diagonal_(0)
    #print("Non-zero: ", torch.nonzero(cos_similarities).size(0))
    cos_similarities = cos_similarities.sum(dim=1)/(feat.size(0)-1)
    #print("node_similarities", cos_similarities)
    nan_mask = torch.isnan(cos_similarities)
    cos_similarities = torch.where(nan_mask, 0, cos_similarities)
    graph_cos_similarities = cos_similarities.sum(dim = 0)/feat.size(0)
    return graph_cos_similarities

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
# def adj_mask_func(a, b):
#     #print(a.nnz, b.nnz)
#     mask = np.array(b.todense(), dtype = bool)
#     a = a.todense()
#     a = np.where(mask, 0, a)
#     a = sp.csr_matrix(a)
#     #print(a.nnz)
#     return a

def adj_mask_func(a, b):
   a_old = a._nnz()
   b_old = b._nnz()
   mask = b.to(torch.bool).to_dense()
   a = a.to_dense()
   adj_mask = torch.where(mask, 0, a)
   # ###### add self loop#######
   # indices = torch.arange(adj_mask.size(0))
   # adj_mask[indices, indices] = torch.where(adj_mask[indices, indices] == 0, 1, adj_mask[indices, indices])
   # ###### add self loop#######
   adj_mask = adj_mask.to_sparse().coalesce()
   mask_per = (a_old - adj_mask._nnz())/a_old*100
   print("mask percent {:1f}%, current nnz {:}".format(mask_per, adj_mask._nnz()))
   
   
   # a = SparseTensor(row=a.coalesce().indices()[0], col=a.coalesce().indices()[1], value = a.coalesce().values(), sparse_sizes=a.size())
   # b = SparseTensor(row=b.coalesce().indices()[0], col=b.coalesce().indices()[1], value = b.coalesce().values(), sparse_sizes=b.size())
   # rowA, colA, _ = a.coo()
   # rowB, colB, _ = b.coo()
   # indexA = rowA * a.size(1) + colA
   # indexB = rowB * a.size(1) + colB
   # nnz_mask = ~(torch.isin(indexA, indexB))
   # a_new = a.masked_select_nnz(nnz_mask)
   # mask_per = (a.nnz() - a_new.nnz())/a.nnz()*100
   # rowA, colA, _ = a_new.coo()
   # indices = torch.stack((rowA, colA))
   # values = a_new.storage._value
   # shape = torch.Size((a_new.size(0),a_new.size(0)))
   # a_new = None
   # adj_mask = torch.sparse.FloatTensor(indices, values, shape)
   # mask_per = (a_old - adj_mask._nnz())/a_old*100
   # print("mask percent {:1f}%, current nnz {:}".format(mask_per, adj_mask._nnz()))
   return adj_mask, mask_per

def adj_mask_v1_func(a, b):
   a_old = a._nnz()
   b_old = b._nnz()
   adj_mask = (a-b).coalesce()
   # adj_mask = torch.where(adj_mask.to_dense()<0, 0, adj_mask.to_dense())
   # adj_mask = adj_mask.to_sparse().coalesce()
   #arr = torch.eq(adj_mask.to_dense(), -1)
   # if torch.any(adj_mask.to_dense()<0):
   #    print("#####assert#####")
   # print(int(torch.sum(a.coalesce().values()).item()), int(torch.sum(b.coalesce().values()).item()), int(torch.sum(adj_mask.coalesce().values()).item()))
   mask_per = (a_old - adj_mask._nnz())/a_old*100
   print("mask percent {:1f}%, current nnz {:}".format(mask_per, adj_mask._nnz()))
   return adj_mask, mask_per

def aug_normalized_adjacency(adj, r):
   t = perf_counter()
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   # d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   # result = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
   d_inv_sqrt = np.power(row_sum, r-1).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   d_right_sqrt = np.power(row_sum, -r).flatten()
   d_right_sqrt[np.isinf(d_right_sqrt)] = 0.
   d_mat_right_sqrt = sp.diags(d_right_sqrt)
   result = d_mat_inv_sqrt.dot(adj).dot(d_mat_right_sqrt).tocoo()
   time = perf_counter() - t
   print("normalize time: ", time)
   return result

def mean_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0]) ##对于SIGN可加可不加
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -1.0).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).tocoo()

##先AAAA，再去冗余，再正则化
def aug_normalized_adjacency_multi(adj, num_hops):
   adj_normalize_buffer = []
   adj_raw = adj + sp.eye(adj.shape[0])
   adj_raw = sparse_mx_to_torch_sparse_tensor(adj_raw).to('cuda:0')
   adj = adj_raw
   adj_dict = {}
   adj_dict['hop_1'] = adj_raw
   adj_mask = adj_raw
   for i in range(1, num_hops+1):
      if i != 1:
         adj = adj_raw.matmul(adj)   ###AAAAA
         adj_mask = adj ###AAA to be masked
         mask_per_total = 0
         for name, mask_adj in adj_dict.items():
            #print("current adj is {:}, || masked adj is {:}".format(f'hop_{i}', name))
            adj_mask, mask_per = adj_mask_func(adj_mask, mask_adj)
            mask_per_total = mask_per_total + mask_per
         print("current hop {:}, mask percent {:1f}%".format(i, mask_per_total))
         adj_dict[f'hop_{i}']= adj_mask
         #print(adj_mask._nnz())
      # adj_mask._values().fill_(1)
      row_sum = adj_mask.to_dense().sum(dim=1)
      d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
      d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
      d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
      adj_norm = d_mat_inv_sqrt.matmul(adj_mask).matmul(d_mat_inv_sqrt)
      #adj_norm = F.normalize(adj_mask.to_dense(), p=2, dim=1).to_sparse()
      adj_normalize_buffer.append(adj_norm) ###A, mask(AA), mask(AAA), ...
   return adj_normalize_buffer

# # #先AAAA，再正则化，再去冗余
# def aug_normalized_adjacency_multi(adj, num_hops):
#    adj_normalize_buffer = []
#    # adj_raw = adj + sp.eye(adj.shape[0])
#    # adj_raw = sparse_mx_to_torch_sparse_tensor(adj_raw).to('cuda:2')
#    # row_sum = adj_raw.cpu().to_dense().sum(dim=1).to('cuda:2')
#    # d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
#    # d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
#    # d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
#    # adj_norm = d_mat_inv_sqrt.matmul(adj_raw).matmul(d_mat_inv_sqrt)
#    adj_norm = adj
#    adj = None
#    adj_dict = {}
#    adj_dict['hop_1'] = adj_norm
#    adj_buffer = adj_norm
#    adj_normalize_buffer = []
#    # adj_normalize_buffer.append(adj_buffer)
#    for i in range(2, num_hops+1):
#       adj_buffer = adj_norm.matmul(adj_buffer)   ###A^A^A^A^A^
#       adj_mask = adj_buffer.cpu()
#       mask_per_total = 0
#       for name, mask_adj in adj_dict.items():
#          #print("current adj is {:}, || masked adj is {:}".format(f'hop_{i}', name))
#          adj_mask, mask_per = adj_mask_func(adj_mask.cpu(), mask_adj.cpu())
#          mask_per_total = mask_per_total + mask_per
#       print("current hop {:}, mask percent {:1f}%".format(i, mask_per_total))
#       adj_dict[f'hop_{i}']= adj_mask
#       # print(adj_mask._nnz())
#       # adj_normalize_buffer.append(adj_mask) ###A, mask(AA), mask(AAA), ...
#    return adj_normalize_buffer

def mean_normalized_adjacency_multi(adj, num_hops):
   adj_normalize_buffer = []
   adj_raw = adj + sp.eye(adj.shape[0])
   adj_raw = sparse_mx_to_torch_sparse_tensor(adj_raw).to('cuda:0')
   adj = adj_raw
   adj_dict = {}
   adj_dict['hop_0'] = adj
   adj_mask = adj
   for i in range(num_hops):
      if i != 0:
         adj = adj_raw.matmul(adj)
         adj_mask = adj
         mask_sum = 0
         print("#", i)
         for name, mask_adj in adj_dict.items():
            #print("adj to be masked", name)
            adj_mask = adj_mask_func(adj_mask, mask_adj)
         adj_dict[f'hop_{i}']= adj_mask
         print(adj_mask._nnz())
      #row_sum = torch.sparse.sum(adj, dim=1).coalesce().values()
      row_sum = adj_mask.to_dense().sum(dim=1)

      # d_inv_sqrt = np.power(row_sum, -0.5).flatten()
      # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
      # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
      # adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
      # adj_normalize_buffer.append(adj_norm)

      d_inv_sqrt = torch.pow(row_sum, -1.0).flatten()
      d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
      d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
      adj_norm = d_mat_inv_sqrt.matmul(adj_mask)
      adj_normalize_buffer.append(adj_norm)
   return adj_normalize_buffer
# #先AAAA，再正则化,再去冗余
# def aug_normalized_adjacency_multi(adj, num_hops):
#    adj_normalize_buffer = []
#    adj_raw = adj + sp.eye(adj.shape[0])
#    adj_raw = sparse_mx_to_torch_sparse_tensor(adj_raw).to('cuda:0')
#    adj = adj_raw
#    adj_dict = {}
#    adj_dict['hop_0'] = adj
#    adj_mask = adj
#    for i in range(num_hops):
#       adj = adj_raw.matmul(adj)
#       adj_mask = adj
#       row_sum = adj_mask.to_dense().sum(dim=1)
#       d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
#       d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
#       d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
#       adj_norm = d_mat_inv_sqrt.matmul(adj_mask).matmul(d_mat_inv_sqrt)
#       adj_mask = adj_norm
#       for name, mask_adj in adj_dict.items():
#          #print(f'hop_{i}', name)
#          adj_mask = adj_mask(adj_mask, mask_adj)
#       adj_dict[f'hop_{i}']= adj_mask
#       adj_normalize_buffer.append(adj_mask)
#    return adj_normalize_buffer
# def aug_normalized_adjacency_multi(adj, num_hops):
#    adj_normalize_buffer = []
#    adj_raw = adj + sp.eye(adj.shape[0])
#    adj_raw = sparse_mx_to_torch_sparse_tensor(adj_raw).to('cuda:0')
#    adj = adj_raw
#    adj_dict = {}
#    adj_dict['hop_0'] = adj
#    for i in range(num_hops):
#       if i != 0:
#          adj = adj_raw.matmul(adj)
#          for name, mask_adj in adj_dict.items():
#             #print(f'hop_{i}', name)
#             adj = adj_mask(adj, mask_adj)
#          adj_dict[f'hop_{i}']= adj
#       # adj = sp.coo_matrix(adj)
#       # row_sum = np.array(adj.sum(1))
#       row_sum = adj.to_dense().sum(dim=1)
      
#       # d_inv_sqrt = np.power(row_sum, -0.5).flatten()
#       # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#       # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#       # adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
#       # adj_normalize_buffer.append(adj_norm)

#       d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
#       d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
#       d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
#       adj_norm = d_mat_inv_sqrt.matmul(adj).matmul(d_mat_inv_sqrt)
#       adj_normalize_buffer.append(adj_norm)
#    return adj_normalize_buffer


def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'Mean': mean_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def fetch_normalization_multi(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency_multi,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'Mean': mean_normalized_adjacency_multi,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
