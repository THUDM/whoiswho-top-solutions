# Top 3 Solution (Team Name: liub)

## Environment
- Python 3.7
- ```pip install -r requirements.txt```

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
```bash
python train.py
python predict.py
```

The output files are in `./saveResult/` folder.
