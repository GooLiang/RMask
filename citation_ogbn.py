import argparse
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
OMP_NUM_THREADS=1
import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
from dataset import load_dataset
from args_ogbn import get_citation_args
from utils import *
from models import get_model

    
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

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


def prepare_data(dataset, normalization="AugNormAdj", num_hops=2, num_wks = 1, cuda=True, model='SGC', device = f"cuda:{1}", seed = 1, r = 0.5):

    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_dataset(dataset, model, device)
    g, adj_raw, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data  #n_classes = labels.max().item()+1
    # t = perf_counter()
    # for i in range(10):
    #     features = torch.spmm(adj_raw, features)   ### AAX     (AX+A2X)/2
    # precompute_time = perf_counter()-t
    if args.model.startswith('RW'):
        t = perf_counter()
        nodes = [i for i in range(g.num_nodes())]
        adj_matrix_rw_total = []
        for i in range(num_wks):
            output_buffer = []
            for hop in range(1, num_hops + 1):
                result = []
                a = dgl.sampling.random_walk(g, nodes, length=hop)
                result.append(nodes)
                result.append(a[0][:,-1].tolist())
                output = index_to_torch_sparse(result)
                output_buffer.append(output)
            adj_matrix_rw_total.append(output_buffer)
        output= []
        for i in range(num_hops):
            sum_output = adj_matrix_rw_total[0][i]
            for j in range(1, num_wks):
                sum_output += adj_matrix_rw_total[j][i]
            sum_output = sum_output.coalesce()
            sum_output = sum_output/num_wks
            output.append(sum_output)
        adj2_list = output
        features_1 = features
    else:
        t = perf_counter()
        adj2_list = neighbor_average_features(g, num_hops)
        features_1 = features
    if cuda:
        features_1 = features_1.to(device)
        adj2_list_final = []
        for adj2 in adj2_list:
            adj2 = adj2.to(device)
            adj2_list_final.append(adj2)
    labels = labels.to(device)
    # move to device
    train_index = train_nid.to(device)
    val_index = val_nid.to(device)
    test_index = test_nid.to(device)
    return adj2_list_final, features_1, labels, train_index, val_index, test_index, evaluator

def train(model, feats, labels, loss_fcn, optimizer, train_loader, RW_model):
    model.train()
    device = labels.device
    if RW_model == 'RW_SIGN' or RW_model == 'SIGN' or RW_model == 'RW_GAMLP' or RW_model == 'GAMLP':
        for batch in train_loader:
            batch_feats = [x[batch].to(device) for x in feats]
            loss = loss_fcn(model(batch_feats), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if RW_model == 'RW_SSGC_large' or RW_model == 'SSGC_large' or RW_model == 'RW_GBP' or RW_model == 'GBP':
        for batch in train_loader:
            batch_feats = feats[batch]
            loss = loss_fcn(model(batch_feats), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(
    model, feats, labels, test_loader, evaluator, train_nid, val_nid, test_nid, RW_model
):
    model.eval()
    device = labels.device
    preds = []
    if RW_model == 'RW_SIGN' or RW_model == 'SIGN' or RW_model == 'RW_GAMLP' or RW_model == 'GAMLP':
        for batch in test_loader:
            batch_feats = [feat[batch].to(device) for feat in feats]
            preds.append(torch.argmax(model(batch_feats), dim=-1))
    if RW_model == 'RW_SSGC_large' or RW_model == 'SSGC_large' or RW_model == 'RW_GBP' or RW_model == 'GBP':
        for batch in test_loader:
            batch_feats = feats[batch]
            preds.append(torch.argmax(model(batch_feats), dim=-1))
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    train_res = evaluator(preds[train_nid], labels[train_nid])
    val_res = evaluator(preds[val_nid], labels[val_nid])
    test_res = evaluator(preds[test_nid], labels[test_nid])
    return train_res, val_res, test_res


def run(args, data, device):
    (
        feats,
        labels,
        in_size,
        num_classes,
        train_nid,
        val_nid,
        test_nid,
        evaluator,
    ) = data
    train_loader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.arange(labels.shape[0]),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Initialize model and optimizer for each run
    model = get_model(args, args.model, in_size, num_classes, args.degree + 1, args.ff_layer, args.input_dropout, args.hidden, args.dropout, args.use_weight, args.cuda, device)

    print("# Params:", get_n_params(model))
    if args.model != 'RW_SSGC_large':
        loss_fcn = nn.CrossEntropyLoss()
    if args.model == 'RW_SSGC_large':
        loss_fcn = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    acc_time = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(model, feats, labels, loss_fcn, optimizer, train_loader, args.model)
        end = time.time()
        acc_time = acc_time + (end - start)
        if epoch % args.eval_every == 0:
            with torch.no_grad():
                acc = test(
                    model,
                    feats,
                    labels,
                    test_loader,
                    evaluator,
                    train_nid,
                    val_nid,
                    test_nid,
                    args.model
                )
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, acc_time)
            acc_time = 0
            log += "Acc: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(*acc)
            print(log)
            if acc[1] > best_val:
                best_epoch = epoch
                best_val = acc[1]
                best_test = acc[2]
            if epoch - best_epoch > args.patience: 
                break
    print(
        "Best Epoch {}, Val {:.4f}, Best Test {:.4f}".format(
            best_epoch, best_val, best_test
        )
    )
    return best_val, best_test

def main(args):
    set_seed(args.seed, args.cuda)
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    with torch.no_grad():
        adj2_list_final, features_raw, labels, idx_train, idx_val, idx_test, evaluator = prepare_data(args.dataset, args.normalization, args.degree,  args.walks, args.cuda, args.model, device, args.seed, args.r)
        if args.model == "RW_SIGN":features, precompute_time, sub_results = sign_mask_precompute(features_raw, adj2_list_final, args.use_weight)
        if args.model == "RW_SSGC_large": features, precompute_time = ssgc_mask_precompute(features_raw, adj2_list_final, args.use_weight)
        if args.model == "RW_GBP": features, precompute_time = gbp_mask_precompute(features_raw, adj2_list_final, args.alpha)
        if args.model == "RW_GAMLP": features, precompute_time, sub_results = sign_mask_precompute(features_raw, adj2_list_final, args.use_weight)
        # print("precompute time {:.4f}s".format(precompute_time))
        if args.model == "GBP":
            emb = adj2_list_final[0]*args.alpha
            for i in range(1, args.degree+1):
                    w_dynamic = args.alpha * math.pow(1-args.alpha, i)
                    emb = emb + w_dynamic * adj2_list_final[i]
            features = emb
        if args.model == "SSGC_large":
            alpha = 0.05
            emb = alpha * features_raw
            for i in range(args.degree):
                    emb = emb + (1-alpha)*adj2_list_final[i]/args.degree
            features = emb
        if args.model == "SIGN" or args.model == "GAMLP": 
            features = adj2_list_final
            adj2_list_final = None
        data = features, labels, features_raw.size(1), labels.max().item()+1, idx_train, idx_val, idx_test, evaluator
        
    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        best_val, best_test = run(args, data, device)
        val_accs.append(best_val)
        test_accs.append(best_test)

    print(
        f"Average val accuracy: {np.mean(val_accs):.4f}, "
        f"std: {np.std(val_accs):.4f}"
    )
    print(
        f"Average test accuracy: {np.mean(test_accs):.4f}, "
        f"std: {np.std(test_accs):.4f}"
    )

if __name__ == "__main__":
    args = get_citation_args()
    print(args)
    print("seed:", args.seed, " hop:", args.degree," walks:", args.walks)
    main(args)
