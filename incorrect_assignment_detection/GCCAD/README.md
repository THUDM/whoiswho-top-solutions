# GCCAD

**Source repo:** https://github.com/THUDM/GraphCAD/

## Environment

```
Python: 3.11.0
CUDA Version: 12.0
torch: 2.1.1
torch_geometric: 2.4.0
pyro-ppl: 1.8.6
```

## Running Steps

``` 
python encoding.py --path your_pub_dir
python build_graph.py --train_dir your_train_dir --test_dir your_test_dir --eval_dir your_eval_dir --pub_dir your_pub_dir
python train.py 
```

- build_graph.py and encoding.py in GCN and GCCAD are the same
- Output files are in res.json
