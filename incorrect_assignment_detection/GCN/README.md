# GCN 

## Running Steps

``` 
python encoding.py --path your_pub_dir

python build_graph.py --author_dir train_author_path --pub_dir train_pub_path --saved_dir save_path --embeddings_dir embedding_path

python build_graph.py --author_dir test_author_path --pub_dir test_pub_path --saved_dir save_path --embeddings_dir embedding_path

python train.py 
```

- build_graph.py and encoding.py in GCN and GCCAD are the same


