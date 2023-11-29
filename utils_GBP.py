import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import f1_score

def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro

def train(model, loader, optimizer, loss_fn):
    model.train()
    loss_list = []
    time_epoch = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        # batch_x = batch_x.cuda(args.dev)
        # batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_train.item())
    return np.mean(loss_list), time_epoch

def validate(model, features, labels):
    model.eval()
    with torch.no_grad():
        output = model(features)
        micro_val = muticlass_f1(output, labels)
        return micro_val.item()

def test_GBP(model, checkpt_file, features_test, labels_test):
    model.load_state_dict(torch.load(f'{checkpt_file}.pkl'))
    model.eval()
    with torch.no_grad():
        output = model(features_test)
        micro_test = muticlass_f1(output, labels_test)
        return micro_test.item()

def train_GBP(epochs, model, checkpt_file, loss_fn, loader, optimizer, features_val, labels_val, patience):
    train_time = 0
    bad_counter = 0
    best = 0
    best_epoch = 0
    best_model = None
    for epoch in range(epochs):
        loss_tra,train_ep = train(model, loader, optimizer, loss_fn)
        with torch.no_grad():
            f1_val = validate(model, features_val, labels_val)
            train_time+=train_ep
            if(epoch+1)%10 == 0: 
                print('Epoch:{:04d}'.format(epoch+1),
                    'train',
                    'loss:{:.3f}'.format(loss_tra),
                    '| val',
                    'acc:{:.3f}'.format(f1_val),
                    '| cost{:.3f}'.format(train_time))
            if f1_val > best:
                best = f1_val
                best_epoch = epoch
                best_model = model
                torch.save(model.state_dict(), f'{checkpt_file}.pkl')
                # print("best_model", best_model.state_dict())
                # print("model", model.state_dict())
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience:
                print("best epoch: ", best_epoch)
                break
    return model, checkpt_file, best, train_time