# Top 2 Solution (Team Name: Complex808)

## Environment
- python3.7
- jupyter notebook环境
- genism==3.4.0 
- numpy==1.20.3 
- scikit-learn==0.19.2

## Data Preparation
Download raw data from http://whoiswho.biendata.xyz/#/data. Make sure that the structure of `data` directory in `whoiswho-top-solutions/from_scratch` is as follows.

```
data
├── sna_test
│   ├── sna_test_pub.json
│   └── sna_test_raw.json
├── sna_valid
│   ├── sna_valid_example.json
│   ├── sna_valid_ground_truth.json
│   ├── sna_valid_pub.json
│   └── sna_valid_raw.json
└── train
    ├── train_author.json
    └── train_pub.json
```

## Running Steps

Run `sna.ipynb` cell by cell.  

Output file `result_test.json` is in `genetest` folder.

 
