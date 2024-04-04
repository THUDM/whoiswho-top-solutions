# GCN 

## Running Steps

``` 
python encoding.py --path your_pub_dir --save_path embedding_save_path

python build_graph.py --author_dir train_author_path --pub_dir your_pub_dir --save_dir save_path --embeddings_dir embedding_save_path

python build_graph.py --author_dir test_author_path --pub_dir your_pub_dir --save_dir save_path --embeddings_dir embedding_save_path

python train.py --train_dir train.pkl --test_dir valid.pkl
```

- build_graph.py and encoding.py in GCN and GCCAD are the same


