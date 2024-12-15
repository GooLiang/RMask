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
from args_ogbn_sign import get_citation_args
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
    data = load_dataset(dataset, device)
    g, adj_raw, features, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data  #n_classes = labels.max().item()+1
    if args.model.startswith('RW'):
        nodes = [i for i in range(g.num_nodes())]
        adj2_list = [None] * (num_hops+1)
        for i in range(num_wks):
            t = perf_counter()
            result = []
            # sum_output[1] = output
            for hop in range(1, num_hops + 1):
                result = []
                a = dgl.sampling.random_walk(g, nodes, length=hop)
                result.append(nodes)
                # result.append(torch.clamp(a[0][:,-1], min=0).tolist())
                result.append(a[0][:,-1].tolist())   ##这里必须加自环，否则找不到hop数的点会返回-1
                output = index_to_torch_sparse(result)
                if i == 0:
                    adj2_list[hop] = output
                else:
                    adj2_list[hop] = adj2_list[hop] + output
            pre_time = perf_counter() - t
            print("finish one round")
            print("Walk one round time: {:.4f}s".format(pre_time))
        output= []
        ##### output.append(adj_raw) #加hop=0的矩阵(错误，本身的RW就已经做了hop=1,不存在hop=0这一概念) 
        # for i in range(1, num_hops + 1):  #
        #     mean_result = sum_output[i]/num_wks
        #     sum_output[i] = None
        #     output.append(mean_result)
        # adj2_list = adj2_list[1:]
        features_1 = features
        # adj2_list = None
        # features_1 = features
    # adj2_list, features_1 = preprocess_citation_RW(adj_raw, features.numpy(), num_hops, num_wks, device, seed, dataset)
    # features_1 = torch.FloatTensor(np.array(features_1)).float()
    else:
        # adj2_list = neighbor_average_features(g, num_hops)
        adj2_list = adj_raw
        features_1 = features
    t = perf_counter()
    cuda = False
    if cuda:
        features_1 = features.to(device)
        # adj2_list_final = adj2_list.to(device)
        adj2_list_final = []
        for adj2 in adj2_list:
            adj2 = adj2.to(device)
            adj2_list_final.append(adj2)
    else:
        # adj2_list_final = adj2_list
        features_1 = features
    # in_feats = g.ndata["feat"].shape[1]
    # feats = neighbor_average_features(g, args)
    pre_time = perf_counter() - t
    print("Pre_trans time: {:.4f}s".format(pre_time))
    labels = labels.to(device)
    # move to device
    train_index = train_nid.to(device)
    val_index = val_nid.to(device)
    test_index = test_nid.to(device)
    # return adj2_list[1:], features_1, labels, n_classes, train_index, val_index, test_index, evaluator
    return adj2_list, features_1, labels, n_classes, train_index, val_index, test_index, evaluator

def train(model, feats, labels, loss_fcn, optimizer, train_loader, RW_model, device, evaluator):
    model.train()
    y_true=[]
    y_pred=[]
    if RW_model == 'RW_SIGN' or RW_model == 'RW_GAMLP' or RW_model == 'GAMLP':
        for batch in train_loader:
            batch_feats = [x[batch].to(device) for x in feats]
            output_att = model(batch_feats)
            y_true.append(labels[batch].to(torch.long))
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            loss = loss_fcn(output_att, labels[batch].to(torch.long).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if RW_model == 'RW_SSGC_large' or RW_model == 'RW_GBP' or RW_model == 'GBP' or RW_model == 'SSGC':
        time_sum = 0
        t = perf_counter()
        for batch in train_loader:
            time_sum = time_sum + (perf_counter()-t)
            batch_feats = feats[batch].to(device)
            t = perf_counter()
            output_att = model(batch_feats)
            y_true.append(labels[batch].to(torch.long))
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            loss = loss_fcn(output_att, labels[batch].to(torch.long).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("train each model time {:.4f}".format(perf_counter()-t))
            t = perf_counter()
        # print("train load_batch sum_time {:.4f}".format(time_sum))
    acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    return acc

def test(
    model, feats, labels, test_loader, evaluator, RW_model, device):
    model.eval()
    preds = []
    true = []
    if RW_model == 'RW_SIGN' or RW_model == 'RW_GAMLP' or RW_model == 'GAMLP':
        for batch in test_loader:
            batch_feats = [feat[batch].to(device) for feat in feats]
            preds.append(torch.argmax(model(batch_feats), dim=-1))
            true.append(labels[batch])
    if RW_model == 'RW_SSGC_large' or RW_model == 'RW_GBP' or RW_model == 'GBP' or RW_model == 'SSGC':
        for batch in test_loader:
            batch_feats = feats[batch].to(device)
            preds.append(torch.argmax(model(batch_feats), dim=-1))
            true.append(labels[batch])
    # Concat mini-batch prediction results along node dimension
    true=torch.cat(true)
    preds = torch.cat(preds, dim=0)
    res = evaluator(preds, true)
    return res

def run(args, model, data, device):
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
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False   #train_nid
    )
    # test_loader = torch.utils.data.DataLoader(
    #     torch.arange(len(train_nid)+len(val_nid),len(train_nid)+len(val_nid)+len(test_nid)), #torch.arange(labels.shape[0]),   
    #     batch_size=args.eval_batch_size,
    #     shuffle=False,
    #     drop_last=False,
    # )
    val_loader = torch.utils.data.DataLoader(
        val_nid, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_nid, batch_size=args.batch_size,
        shuffle=False, drop_last=False)
    print("# Params:", get_n_params(model))
    if args.model != 'RW_SSGC_large':
        loss_fcn = nn.CrossEntropyLoss()
    if args.model == 'RW_SSGC_large':
        loss_fcn = nn.NLLLoss()
    else:
        loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    start = perf_counter()
    # feats = feats.to(device)
    for epoch in range(1, args.epochs + 1):
        t = perf_counter()
        train_acc = train(model, feats, labels, loss_fcn, optimizer, train_loader, args.model, device, evaluator)
        # print("train each epoch time {:.4f}".format(perf_counter()-t))
        if epoch % args.eval_every == 0:
            # t = perf_counter()
            with torch.no_grad():
                val_acc = test(model, feats, labels, val_loader, evaluator, args.model, device)
            # print("val each epoch time {:.4f}".format(perf_counter()-t))
            end = perf_counter()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            start = perf_counter()
            log += "Acc: Train {:.4f}, Val {:.4f}".format(train_acc, val_acc)
            print(log)
            if val_acc > best_val:
                best_epoch = epoch
                best_val = val_acc
                best_test = test(model, feats, labels, test_loader, evaluator, args.model, device)
                print("Epoch {}, Test acc (s): {:.4f}".format(epoch, best_test))

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
    # if args.gpu < 0:
    #     device = "cpu"
    # else:
    device = "cuda:{}".format(args.gpu)

    import torch.optim as optim
    from metrics import accuracy
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
            model.train()
            optimizer.zero_grad()
            output = model(train_features)
            acc_train = accuracy(output, train_labels)
            loss_train = F.cross_entropy(output, train_labels)
            loss_train.backward()
            optimizer.step()
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
    with torch.no_grad():
        adj2_list_final, features, labels, n_classes, idx_train, idx_val, idx_test, evaluator = prepare_data(args.dataset, args.normalization, args.degree,  args.walks, args.cuda, args.model, device, args.seed, args.r)
        if args.model == "RW_SIGN":features, precompute_time, sub_results = sign_mask_precompute(features, adj2_list_final, args.use_weight)
        if args.model == "RW_SSGC_large": features, precompute_time = ssgc_mask_precompute(features, adj2_list_final, args.use_weight)
        if args.model == "RW_GBP": features, precompute_time = gbp_mask_precompute(features, adj2_list_final, args.alpha)
        if args.model == "RW_GAMLP": features, precompute_time, sub_results = sign_mask_precompute(features, adj2_list_final, args.use_weight)
        if args.model == "SSGC": features, precompute_time = ssgc_precompute(features, adj2_list_final, args.degree)
        # print("Total precompute time {:.4f}s".format(precompute_time))
        if args.model == "GBP":
            emb = adj2_list_final[0]*args.alpha
            for i in range(1, args.degree+1):
                    w_dynamic = args.alpha * math.pow(1-args.alpha, i)
                    emb = emb + w_dynamic * adj2_list_final[i]
            features = emb
        if args.model == "SIGN" or args.model == "GAMLP": 
            features = adj2_list_final
            adj2_list_final = None
        data = features, labels, features.size(1), n_classes, idx_train, idx_val, idx_test, evaluator
    
    device = 'cuda:1'
    val_accs = []
    test_accs = []
    model = get_model(args, args.model, features.size(1), n_classes, args.degree + 1, args.ff_layer, args.input_dropout, args.hidden, args.dropout, args.use_weight, args.cuda, device)

    train_time_start = perf_counter() 
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        best_val, best_test = run(args, model, data, device)
        val_accs.append(best_val)
        test_accs.append(best_test)
        # features = features.to(device)
        # model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
        #                 args.epochs, args.weight_decay, args.lr, args.dropout, args.patience)
        # model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
        # with torch.no_grad():
        #     acc_test = test_regression(model, features[idx_test], labels[idx_test])
    train_time = perf_counter() - train_time_start
    print(
        f"Average val accuracy: {np.mean(val_accs):.4f}, "
        f"std: {np.std(val_accs):.4f}"
    )
    print(
        f"Average test accuracy: {np.mean(test_accs):.4f}, "
        f"std: {np.std(test_accs):.4f}"
    )
    print("Total train time {:.4f}s".format(train_time))
if __name__ == "__main__":
    args = get_citation_args()
    print(args)
    print("seed:", args.seed, " hop:", args.degree," walks:", args.walks)
    main(args)
