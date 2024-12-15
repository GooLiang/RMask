import time
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
OMP_NUM_THREADS=1
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
from utils import *
from utils_GBP import *
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def main(args):
    # setting random seeds
    set_seed(args.seed, args.cuda)
    print("lr:",args.lr, "weight-decay:", args.weight_decay, "degree:", args.degree, "num_walks:", args.walks, "seed:", args.seed)
    if args.gpu >= 0:
        device = f"cuda:{args.gpu}"
    else:
        device = 'cpu'
    if args.model.startswith('RW') or args.model.startswith('SGA'):
        print("in citiation")
        adj2_list_final, features, labels, idx_train, idx_val, idx_test = RW_citation(args.dataset, args.normalization, args.degree,  args.walks, args.cuda, args.model, device, args.seed, args.r)
    if args.dataset.startswith('ogbn'):
        if args.dgl:
            print("in ogbn and using dgl")
            adj2_list_final, adj1, features, labels, idx_train, idx_val, idx_test = prepare_data(args.dataset, args.normalization, args.degree,  args.walks, args.cuda, args.model, device, args.seed, args.r, args.dgl)
        else:
            print("in ogbn and using A")
            adj2_list_final, adj1, features, labels, n_classes, idx_train, idx_val, idx_test = prepare_data(args.dataset, args.normalization, args.degree,  args.walks, args.cuda, args.model, device, args.seed, args.r, args.dgl)
    if not (args.model.startswith('RW') and not args.model.startswith('SGA')) and not args.dataset.startswith('ogbn'):
        print("in matmul")
        adj1, features, labels, idx_train, idx_val, idx_test = RW_citation(args.dataset, args.normalization, args.degree,  args.walks, args.cuda, args.model, device, args.seed, args.r)

    print(features.size(1), labels.max().item()+1)
    # device = 'cuda:0'

    if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj1, args.degree)
    if args.model == "SGA_SSGC" or args.model == "RW_SSGC": features, precompute_time = ssgc_mask_precompute(features, adj2_list_final, args.use_weight)
    if args.model == "SGA_SIGN" or args.model == "RW_SIGN": features, precompute_time, sub_results = sign_mask_precompute(features, adj2_list_final, args.use_weight)
    if args.model == "RW_GBP": features, precompute_time = gbp_mask_precompute(features, adj2_list_final, args.alpha)
    if args.model == "SSGC": features, precompute_time = ssgc_precompute(features, adj1, args.degree)
    if args.model == "SIGN" or args.model == "GAMLP": features, precompute_time, sub_results, adj_buffer = sign_precompute(features, adj1, args.degree)
    if args.model == "GBP": features, precompute_time = gbp_precompute(features, adj1, args.degree, args.alpha)
    if args.model == "RW_GAMLP": features, precompute_time, sub_results = sign_mask_precompute(features, adj2_list_final, args.use_weight)
    print("precompute time {:.4f}s".format(precompute_time))
    # device = 'cuda:1'
    # t = perf_counter()
    # features = features.to(device)
    # print("trans time: {:.4f}s".format(perf_counter()-t))
    model = get_model(args, args.model, features.size(1), labels.max().item()+1, args.degree + 1, args.ff_layer, args.input_dropout, args.hidden, args.dropout, args.use_weight, args.cuda, device)
    print("# Params:", get_n_params(model))


    import os
    import uuid
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print('checkpt_file', checkpt_file)
    def train_regression(model,
                        train_features, train_labels,
                        val_features, val_labels,
                        epochs=args.epochs, weight_decay=args.weight_decay,
                        lr=args.lr, dropout=args.dropout, patience = args.patience):
        optimizer = optim.Adam(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
        # scheduler
        if args.scheduler:
            print("--- Use schedular ---")
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
        else:
            scheduler = None

        best_val_loss = 1000000
        best_acc_val = 0
        best_epoch = 0
        best_model = 0
        t = perf_counter()
        for epoch in range(epochs):
            t = perf_counter()
            model.train()
            optimizer.zero_grad()
            output = model(train_features)
            acc_train = accuracy(output, train_labels)
            loss_train = F.cross_entropy(output, train_labels)
            loss_train.backward()
            optimizer.step()
            print("train each epoch time {:.4f}".format(perf_counter()-t))
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                model.eval()
                output = model(val_features)
                loss_val = F.cross_entropy(output, val_labels)
                acc_val = accuracy(output, val_labels)
            if epoch% 10 ==0:
                print("acc_train {:.4f}, acc_val {:.4f}".format(acc_train.item(), acc_val.item()))
            # if loss_val <= best_val_loss:
            #     best_epoch = epoch
            #     best_val_loss = loss_val
            #     torch.save(model.state_dict(), f'{checkpt_file}.pkl')
            if acc_val >= best_acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                # best_model = model
                torch.save(model.state_dict(), f'{checkpt_file}.pkl')
                #print("best epoch", best_epoch)
            if epoch - best_epoch > patience: 
                print("best epoch", best_epoch)
                break
            # print(model.W.weight)
            # print(best_model.W.weight)
        train_time = perf_counter()-t
        return model, best_acc_val, train_time
    
    def test_regression(model, test_features, test_labels):
        model.eval()
        return accuracy(model(test_features), test_labels)

    train_time_start = perf_counter()
    if args.model == "SGC" or args.model == "SSGC" or args.model == 'SGA_SSGC' or args.model == 'RW_SSGC':
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                        args.epochs, args.weight_decay, args.lr, args.dropout, args.patience)
        model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
        with torch.no_grad():
            acc_test = test_regression(model, features[idx_test], labels[idx_test])
    if args.model == "SIGN" or args.model == 'RW_SIGN' or args.model == 'GAMLP' or args.model == 'RW_GAMLP':
        train_batch_feats = [x[idx_train]for x in features]
        val_batch_feats = [x[idx_val]for x in features]
        model, acc_val, train_time = train_regression(model, train_batch_feats, labels[idx_train], val_batch_feats, labels[idx_val],
                        args.epochs, args.weight_decay, args.lr, args.dropout, args.patience)
        model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
        test_batch_feats = [x[idx_test]for x in features]
        with torch.no_grad():
            acc_test = test_regression(model, test_batch_feats, labels[idx_test])
    if args.model.startswith('SGA') and args.use_weight == True:
        input_feats = [feat[idx_train] for feat in features]
        val_feats = [feat[idx_val] for feat in features]
        model, acc_val, train_time = train_regression(model, input_feats, labels[idx_train], val_feats, labels[idx_val],
                        args.epochs, args.weight_decay, args.lr, args.dropout, args.patience)
        input_feats = [feat[idx_test] for feat in features]
        model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
        with torch.no_grad():
            acc_test = test_regression(model, input_feats, labels[idx_test])
    if args.model == "GBP" or args.model == 'RW_GBP':
        loss_fn = nn.CrossEntropyLoss()
        torch_dataset = Data.TensorDataset(features[idx_train], labels[idx_train])
        loader = Data.DataLoader(dataset=torch_dataset,
                                batch_size=args.batch,
                                shuffle=True,
                                num_workers=0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model, checkpt_file, acc_val, train_time = train_GBP(args.epochs, model, checkpt_file, loss_fn, loader, optimizer, features[idx_val], labels[idx_val], args.patience)
        acc_test = test_GBP(model, checkpt_file, features[idx_test], labels[idx_test])
    train_time = perf_counter() - train_time_start

    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # Arguments
    args = get_citation_args()
    print(args)
    if args.tuned:
        if args.model == "SGC" or args.model == 'SGA':
            with open("{}-tuning/{}.txt".format('SGC', args.dataset), 'rb') as f:
                args.weight_decay = pkl.load(f)['weight_decay']
                print("using tuned weight decay: {}".format(args.weight_decay))
        else:
            raise NotImplemented
    main(args)

