### Data
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).

## Compile
g++ -std=c++11 -shared -o rw.so rw.cpp -fPIC

### Usage
For **cora**:
```bash
python citation.py  --model RW_SIGN --degree 8 --dataset cora --weight-decay 1e-5 --lr 0.001 --dropout 0.5 --hidden 512 --epoch 300  --walks 15
python citation.py  --model RW_SSGC --degree 14 --dataset cora --weight-decay 2e-5 --lr 0.5 --dropout 0.5 --hidden 512 --epoch 300  --walks 15
python citation.py  --model RW_GBP --degree 11 --dataset cora --weight-decay 5e-4 --lr 0.01 --alpha 0.1 --dropout 0.5  --hidden 64  --epoch 300 --patience 40  --walks 15 --batch 64
python citation.py  --model RW_GAMLP --degree 8 --dataset citeseer --weight-decay 6e-4 --lr 0.001 --dropout 0.5 --hidden 128 --att-drop 0.5 --pre-process --act leaky_relu --ff-layer 2 --bns --residual --epoch 300  --walks 30 --gpu 1
```

For **ogbn-arxiv**:
```bash
python citation_ogbn.py  --model RW_SIGN --degree 8 --dataset ogbn-arxiv --weight-decay 0 --lr 0.001 --epoch 3000  --walks 30 --seed 1 --gpu 1 --patience 100 --hidden 512 --input-drop 0.2
python citation_ogbn.py  --model RW_GBP --degree 5 --dataset ogbn-arxiv --wght-decay 0 --lr 0.001 --epoch 2000  --walks 30 --seed 1 --gpu 0 --patience 100 --hidden 1024 --eval-every 10 --ff-layer 3 --alpha 0.05 --batch 10000 --drop 0.
python citation_ogbn.py  --model RW_SSGC_large --degree 10 --dataset ogbn-arxiv --wght-decay 0 --lr 0.001 --epoch 2000  --walks 30 --seed 1 --gpu 0 --patience 100 --hidden 1024 --eval-every 10 --ff-layer 2 --alpha 0.05 --batch 10000 --drop 0.5
python citation_ogbn.py  --model RW_GAMLP --degree 10 --dataset ogbn-arxiv --weight-decay 0 --lr 0.001 --dropout 0.5 --hidden 512 --att-drop 0.6 --pre-process --act leaky_relu --bns --ff-layer 2 --epoch 300  --walks 30 --seed 1 --gpu 1
```

For **ogbn-products**:
```bash
python citation_ogbn.py  --model RW_SIGN --degree 8 --dataset ogbn-products --weight-decay 0 --lr 0.001 --epoch 3000  --walks 20 --seed 1 --gpu 4 --patience 200 --hidden 512 --input-drop 0.5
 python citation_ogbn.py  --model RW_SSGC_large --degree 8 --dataset ogbn-products --weight-decay 0 --lr 0.001 --epoch 3000  --walks 20 --seed 2 --gpu 4 --patience 200 --hidden 512 --input-drop 0.5
python citation_ogbn.py --model RW_GBP --degree 5 --dataset ogbn-products --weight-decay 0 --lr 0.01 --alpha 0.05 --dropout 0.5  --hidden 1024  --epoch 3000 --patience 200  --walks 20 --seed 1 --gpu 4
python citation_ogbn.py  --model RW_GAMLP --degree 5 --dataset ogbn-products --weight-deca0 --lr 0.002 --dropout 0.5 --att-drop 0.5 --pre-process --act leaky_relu --ff-layer 4 --bns --residual --epoch 3000  --walks 20 --seed 2 --gpu 5 --patience 200 --hidden 1024 --input-drop 0.2
```