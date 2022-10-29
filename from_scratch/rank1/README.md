# Top 1 Solution (Team Name: ECNU_AIDA)

## Environment
- Python==3.7
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
python save_text.py
python train_w2v.py
python main.py 
```

Output files are in `./res/output/` folder.
