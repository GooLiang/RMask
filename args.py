import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0,   #5e-6   0.3 0
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="RW_SIGN",
                        choices=["SGC", "GCN", "SSGC","SIGN","GBP", "SGA_SSGC", "SGA_SIGN", "RW_GBP", "RW_SIGN", "RW_SSGC", 'RW_SSGC_large', "GAMLP", "RW_GAMLP"],
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj', 'Mean'],
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=8,
                        help='degree of the approximation.')
    parser.add_argument('--walks', type=int, default=20,
                        help='number of thr walks.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='id of gpu.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')
    parser.add_argument('--use-weight', action='store_true', default=False, help='use weight')
    parser.add_argument('--dgl', default=False, help='use dgl for large graph')
    parser.add_argument('--patience', type=int, default=100, help='Patience.')
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99,
                        help="decay the lr by gamma for ExponentialLR scheduler")
    parser.add_argument('--input-dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--ff-layer", type=int, default=2, help="number of feed-forward layers")
    parser.add_argument('--alpha', type=float, default=0.5,   #5e-6   0.3 0
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--r', type=float, default=0.5, help='operator.')
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--act", type=str, default="relu",
                        help="the activation function of the model")
    parser.add_argument("--pre-process", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--pre-dropout", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
